
#------------------------------------------------------------------------------------------------------------#
#                                   Loss and Optimizer Defination
#------------------------------------------------------------------------------------------------------------#
class QVAELossLayer(Layer):
    def __init__(self, alpha, beta, lossFlag, **kwargs):
        self.alpha = alpha
        self.beta = beta
        self.quadraloss = lossFlag
        super(QVAELossLayer, self).__init__(**kwargs)

    def KL_loss(self, inputs, raxis=1):
        kl_anchor = 1 + inputs[3] - tf.keras.backend.square(inputs[0]) - tf.keras.backend.exp(inputs[3])
        kl_positive = 1 + inputs[4] - tf.keras.backend.square(inputs[1]) - tf.keras.backend.exp(inputs[4])
        kl_negative = 1 + inputs[5] - tf.keras.backend.square(inputs[2]) - tf.keras.backend.exp(inputs[5])
        if self.quadraloss:
            kl_negative = kl_negative + 1 + inputs[13] - tf.keras.backend.square(inputs[12]) - \
                          tf.keras.backend.exp(inputs[13])

        kl_loss = kl_anchor + kl_positive + kl_negative
        kl_loss = tf.reduce_sum(kl_loss, axis=raxis)
        kl_loss *= -0.5
        return kl_loss

    def triplet_loss(self, inputs):
        anchor, positive, negative = inputs[0:3]
        p_dist = K.sum(K.square(anchor - positive), axis=-1)
        n_dist = K.sum(K.square(anchor - negative), axis=-1)
        return K.sum(K.maximum(p_dist - n_dist + self.alpha, 0), axis=0)

    def quadruplet_loss(self, inputs):
        ap_dist, an_dist, nn_dist = inputs[-3::]

        # square
        ap_dist2 = tf.keras.backend.square(ap_dist)
        an_dist2 = tf.keras.backend.square(an_dist)
        nn_dist2 = tf.keras.backend.square(nn_dist)

        return K.sum(K.maximum(ap_dist2 - an_dist2 + self.alpha, 0), axis=0) + K.sum(
            K.maximum(ap_dist2 - nn_dist2 + self.beta, 0), axis=0)

    def reconstruction_loss(self, inputs):
        cross_ent = (tf.nn.sigmoid_cross_entropy_with_logits(logits=inputs[6], labels=inputs[9]) +
                     tf.nn.sigmoid_cross_entropy_with_logits(logits=inputs[7], labels=inputs[10]) +
                     tf.nn.sigmoid_cross_entropy_with_logits(logits=inputs[8], labels=inputs[11]))
        if self.quadraloss:
            cross_ent = cross_ent + tf.nn.sigmoid_cross_entropy_with_logits(logits=inputs[14], labels=inputs[15])

        return tf.reduce_sum(cross_ent, axis=[1, 2, 3])

    def call(self, inputs, **kwargs):
        loss1 = self.triplet_loss(inputs)
        if self.quadraloss:
            loss1 = self.quadruplet_loss(inputs)

        loss2 = self.KL_loss(inputs)
        loss3 = self.reconstruction_loss(inputs)
        loss = tf.reduce_mean(0.85*loss1 + 0.15*(loss3 + loss2))
        self.add_loss(loss)
        return loss

