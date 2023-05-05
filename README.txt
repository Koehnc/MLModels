In order to run the trials, you should be able to run Runner.py and GANRunner.py
Runner.py with run the simple NN. Uncomment line 55-57 to run the NE algorithm; change the % operator to control
how often it is trained. It takes a lot of time, so training all 50000 images is not recomended.

To run the GAN, just run the GANRunner.py file.

Changing the structures:
 - NN: line 41, change the structure however
 - NE: line 49, population size is first, structure is second
 - GAN: DCGAN.py includes the instantiation of the generator and discriminator. Change there as wanted