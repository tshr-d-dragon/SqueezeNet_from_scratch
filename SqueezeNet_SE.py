import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from keras.layers import Conv2D, BatchNormalization, Activation, MaxPooling2D, Dropout
from keras.layers import Concatenate, Input, GlobalAveragePooling2D, Dense, Multiply
from keras.models import Model


def SqueezeNet(input_shape, classes):

  
  def SqueezeAndExcitation(inputs, ratio=8):
    
    b, h, w, c = inputs.shape
    
    x = GlobalAveragePooling2D()(inputs)
    x = Dense(c//ratio, activation='relu', use_bias=False)(x)
    x = Dense(c, activation='sigmoid', use_bias=False)(x)
    
    x = Multiply()([inputs, x])
    
    return x

  
  def Fire(inputs, fs, fe):

      s1 = Conv2D(filters=fs, kernel_size=1, padding='same', use_bias=False, activation='relu')(inputs)
      s1 = BatchNormalization()(s1)
      e1 = Conv2D(filters=fe, kernel_size=1, padding='same', use_bias=False, activation='relu')(s1)
      e1 = BatchNormalization()(e1)
      e3 = Conv2D(filters=fe, kernel_size=3, padding='same', use_bias=False, activation='relu')(s1)
      e3 = BatchNormalization()(e3)

      output = Concatenate()([e1, e3])
  
      return output


  inputs = Input(input_shape)
  x1 = Conv2D(filters=96, kernel_size=7, strides=2, padding='same', use_bias=False, activation='relu')(inputs)
  x1 =  BatchNormalization()(x1)
  x1 = MaxPooling2D(pool_size=3, strides=2, padding='same')(x1)

  f2 = Fire(x1, 16, 64)
  f3 = Fire(f2, 16, 64)
  f4 = Fire(f3, 32, 128)
  f4 = SqueezeAndExcitation(f4, ratio=16)
  x1 = MaxPooling2D(pool_size=3, strides=2, padding='same')(f4)

  f5 = Fire(x1, 32, 128)
  f6 = Fire(f5, 48, 192)
  f7 = Fire(f6, 48, 192)
  f8 = Fire(f7, 64, 256)
  f8 = SqueezeAndExcitation(f8, ratio=16)
  x1 = MaxPooling2D(pool_size=3, strides=2, padding='same')(f8)

  f8 = Fire(x1, 64, 256)
  x1 = Conv2D(filters=classes, kernel_size=1)(f8)
  x1 = GlobalAveragePooling2D()(x1)

  if classes == 1:
    x1 = Activation('sigmoid')(x1)
  else:
    x1 = Activation('softmax')(x1)

  model = Model(inputs=inputs, outputs=x1)
  # model.summary()

  return model


if __name__== '__main__':

  model = SqueezeNet(input_shape = (224, 224, 3), classes=1000)
