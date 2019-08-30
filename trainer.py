import numpy as np
from utils import *
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

class Trainer():
  def __init__(self, model, PATH, duration):
    self.model = model
    self.disc_loss = []

    self.disc_loss_r = []
    self.disc_loss_f = []
    
    self.crit_loss_r = []
    self.crit_loss_f = []

    self.crit_loss = []
    self.gen_loss = []
    self.path = PATH
    self.duration = duration
    
  def train_wgan(self, epochs, batch_size, sample_interval, train_data):
    """
    Trains a WGAN
    """

    
    # Create labels for real and fake data
    real = np.ones((batch_size, 1))
    fake = -np.ones((batch_size, 1))

    for epoch in range(epochs):
        for _ in range(self.model.n_critic):

          real_seqs = get_batch(train_data, batch_size)
          noise = np.random.normal(0, 1, (batch_size, self.model.latent_dim))

          # Generate a batch of new note sequences
          gen_seqs = self.model.generator.predict(noise)

          # Train the discriminator
          c_loss_real = self.model.critic.train_on_batch(real_seqs, real)
          c_loss_fake = self.model.critic.train_on_batch(gen_seqs, fake)


          # Clip critic weights
          for l in self.model.critic.layers:
              weights = l.get_weights()
              weights = [np.clip(w, -self.model.clip_value, self.model.clip_value) for w in weights]
              l.set_weights(weights)

        noise = np.random.normal(0, 1, (batch_size, self.model.latent_dim))

        # Train the generator (to have the discriminator label samples as real)
        g_loss = self.model.wgan.train_on_batch(noise, real)

        if epoch % sample_interval == 0:
          print ("%d [CritLoss(Real): %10f] [CritLoss(Fake): %10f] [GenLoss = %10f]" % (epoch, c_loss_real[0], c_loss_fake[0], 100*g_loss))

          self.crit_loss_r.append(c_loss_real)
          self.crit_loss_f.append(c_loss_fake)
          self.gen_loss.append(g_loss)
          sample_image(self.model, epoch, real_seqs, self.path)
        if (epoch % 1000 == 0):
          self.save_models(self.path, epoch, self.model.generator, self.model.critic)
    
    self.savedata(self.path, train_data)
    self.showLoss(self.path, save=True)
  
       
  def train_gan(self, epochs, batch_size, sample_interval, train_data):
    """
    Trains a GAN 
    """
    
    # Create labels for real and fake data
    real = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))

    for epoch in range(epochs):

        # Get batch of real data
        real_seqs = get_batch(train_data, batch_size)
        # Generate batch of fake data using random noise
        noise = np.random.normal(0, 1, (batch_size, self.model.latent_dim))
        gen_seqs = self.model.generator.predict(noise)

        # Train the discriminator to accept real data and reject fake data
        d_loss_real = self.model.discriminator.train_on_batch(real_seqs, real)
        d_loss_fake = self.model.discriminator.train_on_batch(gen_seqs, fake)


        # Train the generator such that when it takes random noise as an
        # input, it will produce fake data which the discriminator accepts
        # as real

        noise = np.random.normal(0, 1, (batch_size, self.model.latent_dim))
        g_loss = self.model.gan.train_on_batch(noise, real)

        if epoch % sample_interval == 0:
          print ("%d [DiscLoss/Acc Real: (%10f, %10f)] [DiscLoss/Acc Fake: (%10f, %10f)] [DiscAcc %10f][GenLoss = %10f]" % (epoch,
                                                                                                             d_loss_real[0], d_loss_real[1], 
                                                                                                             d_loss_fake[0], d_loss_fake[1], 
                                                                                                             0.5 * (d_loss_real[1] + d_loss_fake[1]), 
                                                                                                             g_loss))

          self.disc_loss_r.append(d_loss_real)
          self.disc_loss_f.append(d_loss_fake)

          self.gen_loss.append(g_loss)
          sample_image(self.model, epoch, real_seqs, self.path)
        if (epoch % 1000 == 0):
          self.save_models(self.path, epoch, self.model.generator, self.model.discriminator)
    
    self.savedata(self.path, train_data)
    self.showLoss(self.path, save=True)
    
  def save_models(self, path, epoch, generator, discriminator=None,  critic=None):
      """
      Saves the models
      """
      generator.save('{}/models/generator_{}.h5'.format(path, epoch))
      
      if discriminator:
        discriminator.save('{}/models/discriminator_{}.h5'.format(path, epoch))
      if critic:
        critic.save('{}/models/critic_{}.h5'.format(path, epoch))
      
  def savedata(self, path, train_set):
      """
      Saves the loss data
      """
      print("saving")
      fn1 = path + "/gen_loss.npy"
      
      fn2 = path + "/disc_loss_r.npy"
      fn3 = path + "/disc_loss_f.npy"
      
      fn4 = path + "/crit_loss_r.npy"
      fn5 = path + "/crit_loss_f.npy"

      

      np.save(fn1, self.gen_loss)
      
      np.save(fn2, self.disc_loss_r)
      np.save(fn3, self.disc_loss_f)
      
      np.save(fn4, self.crit_loss_r)
      np.save(fn5, self.crit_loss_f)



      
  def showLoss(self, path, save=True):
    """
    Plots losses from training process
    """
    onRealLoss =  np.array(self.disc_loss_r)[:, 0]
    onRealAcc  =  np.array(self.disc_loss_r)[:, 1]

    onFakeLoss =  np.array(self.disc_loss_f)[:, 0]
    onFakeAcc  =  np.array(self.disc_loss_f)[:, 1]

    g_Loss = np.array(self.gen_loss)
    plt.figure(figsize = (8,5.5), dpi=100)

    plt.title("Discriminator and Generated Losses")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")

    plt.plot([-10,1010], [np.log(2), np.log(2)], '-', label="Expected value (log 2)", c= 'k', lw=3)

    plt.plot(savgol_filter(onRealLoss,11,1), label="Discriminator on Real", c='g')
    plt.plot(savgol_filter(onFakeLoss,11,1), label="Discriminator on Fake", c='r')
    plt.plot(savgol_filter(g_Loss,15,1), label="Generator", c='b')
    plt.grid()
    plt.legend()

    if save==True:
      plt.savefig(path + "/plots/Losses.png", edgecolor='k', dpi=100)


    plt.figure(figsize = (8,6), dpi=100)
    plt.title("Accuracy of Discriminator in Correctly Identifying Real and Fake samples")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")

    plt.plot(savgol_filter(onFakeAcc,51,3), label="Fake", c='r')
    plt.plot(savgol_filter(onRealAcc,51,3), label="Real", c='g')
    plt.plot([-10,1010], [0.5,0.5], '-', label="Expected value (0.5)", c= 'k', lw=3)
    plt.grid()
    plt.legend()
    if save==True:
      plt.savefig(path + "/plots/Accuracies.png", edgecolor='k', dpi=100)
