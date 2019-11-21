import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Input, Lambda, Flatten, Reshape, Conv2D
from tensorflow.keras.callbacks import LambdaCallback
from flows import Flow
from layers import GatedConv2D, GatedConv2DTranspose, VariationalLayer

class GatedConvVAE(tf.Module):
    def __init__(self, img_wt, img_ht, flow: Flow = None, hidden_units=32, z_size=64, callbacks=[], metrics=[],
                 output_activation='sigmoid', loss='binary_crossentropy', beta_update_fn=None,
                 min_beta=0.01, max_beta=1):
        super(GatedConvVAE, self).__init__()
        if beta_update_fn is None:
            beta_update_fn = lambda i, beta: 1.0E-2*(i+1)
        self.flow = flow
        self.hidden_units = hidden_units
        self.z_size = z_size
        self.output_activation = output_activation
        self.min_beta = tf.constant(min_beta)
        self.max_beta = tf.constant(max_beta)
        self.encoder = self._create_encoder(img_wt, img_ht)
        self.decoder, self.variational_layer = self._create_decoder(img_wt, img_ht)
        beta_update = LambdaCallback(on_epoch_begin=lambda i,_: beta_update_fn(i, self.variational_layer.beta))
        self.model = Model(inputs=self.encoder.inputs, outputs=self.decoder(self.encoder(self.encoder.inputs)))
        self.model.compile(loss=loss, optimizer='adam', callbacks=[beta_update]+callbacks, metrics=metrics)

    def fit(self, *args, **kwargs):
        return self.model.fit(*args, **kwargs)

    def predict(self, *args, **kwargs):
        return self.model.predict(*args, **kwargs)

    def evaluate(*args, **kwargs):
        return self.model.evaluate(*args, **kwargs)

    def _conv_downsample(self, f, x):
        g = GatedConv2D(f, 3, activation='linear')
        g_downsample = GatedConv2D(f, 3, strides=2)
        return g_downsample(g(x))

    def _conv_upsample(self, f, x):
        g = GatedConv2DTranspose(f, 3, activation='linear')
        g_upsample = GatedConv2DTranspose(f, 3, strides=2)
        return g_upsample(g(x))

    def _create_encoder(self, wt, ht):
        input_0 = Input((wt, ht, 1))
        h_conv = self._conv_downsample(self.hidden_units, input_0)
        h = self._conv_downsample(self.hidden_units*2, h_conv)
        z_mu = Dense(self.z_size, activation='linear')(Flatten()(h))
        z_log_var = Dense(self.z_size, activation='linear')(Flatten()(h))
        outputs = [z_mu, z_log_var]
        if self.flow is not None:
            params = Dense(self.flow.param_count(self.z_size), activation='linear')(Flatten()(h))
            outputs += [params]
        return Model(inputs=input_0, outputs=outputs)


    def _create_decoder(self, wt, ht):
        z_mu = Input(shape=(self.z_size,))
        z_log_var = Input(shape=(self.z_size,))
        inputs = [z_mu, z_log_var]
        if self.flow is not None:
            params = Input(shape=(self.flow.param_count(self.z_size),))
            inputs += [params]
        v_layer = VariationalLayer(self.flow)
        z_0, z_k, ldj, kld = v_layer(inputs)
        h_k = Dense(wt//4 * ht//4, activation='linear')(z_k)
        h_k = Reshape((wt//4, ht//4, 1))(h_k)
        h_conv = self._conv_upsample(self.hidden_units*2, h_k)
        x_out = self._conv_upsample(self.hidden_units, h_conv)
        output_0 = Conv2D(1, 1, activation=self.output_activation, padding='same')(x_out)
        return Model(inputs=inputs, outputs=output_0), v_layer
