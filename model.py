#------------------------------------------------------------------------------------------------------------#
# Build Neural Network for computing triplet similarity
#------------------------------------------------------------------------------------------------------------#
class CVAE(tf.keras.Model):
  """Convolutional variational autoencoder."""

  def __init__(self, input_shape, latent_dim):
    super(CVAE, self).__init__()
    self.latent_dim = latent_dim
    self.encoder = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=input_shape),
            tf.keras.layers.Conv2D(
                filters=32, kernel_size=3, strides=(1, 1), activation='relu'),
            tf.keras.layers.Conv2D(
                filters=64, kernel_size=3, strides=(2, 2), activation='relu'),
            tf.keras.layers.Conv2D(
                filters=128, kernel_size=3, strides=(2, 2), activation='relu'),
            tf.keras.layers.Conv2D(
                filters=256, kernel_size=3, strides=(2, 2), activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(4096, activation='relu'),
            # No activation
            tf.keras.layers.Dense(latent_dim + latent_dim),

        ]
    )

    self.decoder = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
            tf.keras.layers.Dense(units=8 * 8 * 256, activation=tf.nn.relu),
            tf.keras.layers.Reshape(target_shape=(8, 8, 256)),
            tf.keras.layers.Conv2DTranspose(
                filters=256, kernel_size=3, strides=2, padding='same',
                activation='relu'),
            tf.keras.layers.Conv2DTranspose(
                filters=64, kernel_size=3, strides=2, padding='same',
                activation='relu'),
            # No activation
            tf.keras.layers.Conv2DTranspose(
                filters=32, kernel_size=3, strides=2, padding='same', activation='relu'),
            tf.keras.layers.Conv2D(
                filters=1, kernel_size=3, strides=1, padding='same'),
        ]
    )

  @tf.function
  def sample(self, eps=None):
    if eps is None:
      eps = tf.random.normal(shape=(100, self.latent_dim))
    return self.decode(eps, apply_sigmoid=True)

  def encode(self, x):
    mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
    return tf.keras.backend.l2_normalize(mean, axis=-1), logvar #tf.keras.activations.relu(logvar)

  def reparameterize(self, mean, logvar):
    eps = eps = tf.random.normal(shape=[mean.shape[-1]], mean=0., stddev=1.0)
    return eps * tf.exp(logvar * .5) + mean

  def decode(self, z, apply_sigmoid=False):
    logits = self.decoder(z)
    if apply_sigmoid:
      probs = tf.sigmoid(logits)
      return probs
    return logits

  def call(self, x):
      mean, _ = self.encode(x)
      return mean


class metricNet(tf.keras.Model):
    def __init__(self, latent_dim):
        super(metricNet, self).__init__()
        self.latent_dim = latent_dim
        self.network = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(latent_dim*2,)),
                tf.keras.layers.Dense(10, activation='relu'),
                tf.keras.layers.Dense(10, activation='relu'),
                tf.keras.layers.Dense(5, activation='relu'),
                tf.keras.layers.Dense(2, activation='softmax'),
                tf.keras.layers.Lambda(lambda x: x[:, 0]),
            ]
        )

    def call(self, x):
        encoded = self.network(x)
        return encoded
        
        
def build_model(input_shape, network, metricnetwork=None, margin=0.1, margin2=0.01, QuadralossEnable = False):
    '''
    Define the Keras Model for training
        Input :
            input_shape : shape of input images
            network : Neural network to train outputing embeddings
            margin : minimal distance between Anchor-Positive and Anchor-Negative for the lossfunction (alpha)

    '''
    # Define the tensors for the three input images
    anchor_input = Input(input_shape, name="anchor_input")
    positive_input = Input(input_shape, name="positive_input")
    negative_input = Input(input_shape, name="negative_input")


    # Generate the encodings (feature vectors) for the three images
    Amean, Alogvar = network.encode(anchor_input)
    Pmean, Plogvar = network.encode(positive_input)
    Nmean, Nlogvar = network.encode(negative_input)


    # reparameterization for latent space
    Az = network.reparameterize(Amean, Alogvar)
    Pz = network.reparameterize(Pmean, Plogvar)
    Nz = network.reparameterize(Nmean, Nlogvar)


    # reconstructed image
    Ax_logit = network.decode(Az)
    Px_logit = network.decode(Pz)
    Nx_logit = network.decode(Nz)



    if QuadralossEnable:
        negative2_input = Input(input_shape, name="negative2_input")
        Nmean2, Nlogvar2 = network.encode(negative2_input)
        Nz2 = network.reparameterize(Nmean2, Nlogvar2)
        Nx_logit2 = network.decode(Nz2)

        # compute the concatenated pairs
        encoded_ap = Concatenate(axis=-1, name="Anchor-Positive")([Amean, Pmean])
        encoded_an = Concatenate(axis=-1, name="Anchor-Negative")([Amean, Nmean])
        encoded_nn = Concatenate(axis=-1, name="Negative-Negative2")([Nmean, Nmean2])

        # compute the distances AP, AN, NN
        ap_dist = metricnetwork.network(encoded_ap)
        an_dist = metricnetwork.network(encoded_an)
        nn_dist = metricnetwork.network(encoded_nn)

        # TripletLoss Layer
        loss_layer = QVAELossLayer(alpha=margin, beta=margin2, lossFlag=QuadralossEnable, name='triplet_loss_layer')(
            [Amean, Pmean, Nmean,
             Alogvar, Plogvar, Nlogvar,
             Ax_logit, Px_logit, Nx_logit,
             anchor_input, positive_input, negative_input,
             Nmean2, Nlogvar2, Nx_logit2, negative2_input,
             ap_dist, an_dist, nn_dist])
        # Connect the inputs with the outputs
        network_train = Model(inputs=[anchor_input, positive_input, negative_input, negative2_input], outputs=loss_layer)

    else:
        # TripletLoss Layer
        loss_layer = QVAELossLayer(alpha=margin, beta=margin2, lossFlag=QuadralossEnable, name='triplet_loss_layer')(
            [Amean, Pmean, Nmean,
             Alogvar, Plogvar, Nlogvar,
             Ax_logit, Px_logit, Nx_logit,
             anchor_input, positive_input, negative_input])
        # Connect the inputs with the outputs
        network_train = Model(inputs=[anchor_input, positive_input, negative_input], outputs=loss_layer)

    # return the model
    return network_train
