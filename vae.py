import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

# cuda setup
device = torch.device("cuda")
kwargs = {'num_workers': 1, 'pin_memory': True} 

# hyper params
batch_size = 64

epochs = 100


train_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('../data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('../data', train=False, transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=False)


def one_hot(labels, class_size):
    targets = torch.zeros(labels.size(0), class_size)
    for i, label in enumerate(labels):
        targets[i, label] = 1
    return targets.to(device)


class Model(nn.Module):
    def __init__(self,latent_size=128,num_classes=10):
        super(Model,self).__init__()
        self.latent_size = latent_size
        self.num_classes = num_classes

        # For encode
        self.conv1 = nn.Conv2d(3+1, 64, kernel_size=3, padding=1, stride=2)
        #self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2)
        #self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2)


        self.linear1 = nn.Linear(4*4*256,1024)
        self.mu = nn.Linear(1024, self.latent_size)
        self.logvar = nn.Linear(1024, self.latent_size)

        # For decoder
        self.linear2 = nn.Linear(self.latent_size + self.num_classes, 1024)
        self.linear3 = nn.Linear(1024,4*4*256)

        self.conv6 = nn.ConvTranspose2d(256, 256, kernel_size=4, padding=1, stride=2)
        self.conv7 = nn.ConvTranspose2d(256, 128, kernel_size=3, padding=1)
        self.conv8 = nn.ConvTranspose2d(128, 128, kernel_size=4, padding=1, stride=2)
        self.conv9 = nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1)
        self.conv10 = nn.ConvTranspose2d(64, 64, kernel_size=4, padding=1, stride=2)
        self.conv11 = nn.ConvTranspose2d(64, 3, kernel_size=3, padding=1)

        self.bn1 = nn.BatchNorm2d(256)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(64)

    def encoder(self,x,y):
        y = torch.argmax(y, dim=1).reshape((y.shape[0],1,1,1))
        y = y.expand(-1, -1, x.size(2), x.size(3))
        t = torch.cat((x,y),dim=1)
        
        t = F.relu(self.bn3(self.conv1(t)))
        #t = F.relu(self.bn3(self.conv2(t)))
        t = F.relu(self.bn2(self.conv3(t)))
        #t = F.relu(self.bn2(self.conv4(t)))
        t = F.relu(self.bn1(self.conv5(t)))

        t = t.reshape((x.shape[0], -1))
        #print(t.shape)
        
        t = F.relu(self.linear1(t))
        mu = self.mu(t)
        logvar = self.logvar(t)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std).to(device)
        return eps*std + mu
    
    def unFlatten(self, x):
        return x.reshape((x.shape[0], 256, 4, 4))

    def decoder(self, z):
        t = F.relu(self.linear2(z))
        t = F.relu(self.linear3(t))
        t = self.unFlatten(t)

        t = F.relu(self.bn1(self.conv6(t)))
        t = F.relu(self.bn2(self.conv7(t)))
        t = F.relu(self.bn2(self.conv8(t)))
        t = F.relu(self.bn3(self.conv9(t)))
        t = F.relu(self.bn3(self.conv10(t)))
        t = F.relu(self.conv11(t))

        return t


    def forward(self, x, y):
        mu, logvar = self.encoder(x,y)
        z = self.reparameterize(mu,logvar)

        # Class conditioning
        z = torch.cat((z, y.float()), dim=1)
        pred = self.decoder(z)
        return pred, mu, logvar
# create a CVAE model

model = Model().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.mse_loss(recon_x, x, reduction='sum')
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, labels) in enumerate(train_loader):
        data, labels = data.to(device), labels.to(device)
        labels = one_hot(labels, 10)
        recon_batch, mu, logvar = model(data, labels)
        optimizer.zero_grad()
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.detach().cpu().numpy()
        optimizer.step()
        if batch_idx % 20 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))


def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, labels) in enumerate(test_loader):
            data, labels = data.to(device), labels.to(device)
            labels = one_hot(labels, 10)
            recon_batch, mu, logvar = model(data, labels)
            test_loss += loss_function(recon_batch, data, mu, logvar).detach().cpu().numpy()
            if i == 0:
                n = min(data.size(0), 10)
                comparison = torch.cat([data[:n],
                                      recon_batch.view(-1, 3, 32, 32)[:n]])
                save_image(comparison.cpu(),
                         'reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))

for epoch in range(1, epochs + 1):
        train(epoch)
        test(epoch)

with torch.no_grad():
    for i in range(10):
        label = torch.zeros(10,10).cuda()
        label[:, i] = 1
        z = torch.randn(10, 128).to(device)
        # z = torch.cat((z, label), dim=1)
        latent = torch.cat((z, label), dim=1)
        output = model.decoder(latent).cpu()
        save_image(output.view(10, 3, 32, 32),
                'sample_' + str(epoch) + '.png')
        
import numpy
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy import asarray
from numpy.random import shuffle
from scipy.linalg import sqrtm
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.datasets.mnist import load_data
from skimage.transform import resize

def scale_images(images, new_shape):
 images_list = list()
 
 for image in images:
  new_image = resize(image, new_shape, 0)
  images_list.append(new_image)
 return asarray(images_list)
 
def calculate_fid(model, images1, images2):
 print("1")
 print(images1)
 print("1")
 print(images2)
 act1 = model.predict(images1)
 act2 = model.predict(images2)
 mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
 mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
 
 ssdiff = numpy.sum((mu1 - mu2)**2.0)
 
 covmean = sqrtm(sigma1.dot(sigma2))
 
 if iscomplexobj(covmean):
  covmean = covmean.real
 
 fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
 return fid
def Fid(gmodel):
  model = InceptionV3(include_top=False, pooling='avg', input_shape=(32,32,3))

  loaderfi = torch.utils.data.DataLoader(train_set, batch_size, shuffle=True,
                           num_workers=4, persistent_workers=True, pin_memory=True)
  for i in loaderfi:
    images1 = torch.tensor(i[0])
    break

  images1 = (images1.clamp(-1, 1) + 1) / 2
  images1 = (images1 * 255).type(torch.uint8)

  images2=model.sample(gmodel, 10000)
  return_image=images2
  images2 = images2.cpu()

  images1 = scale_images(images1, (32,32,3))
  images2 = scale_images(images2, (32,32,3))
  
  images1 = preprocess_input(images1)
  images2 = preprocess_input(images2)

  fid = calculate_fid(model, images1, images2)
  print('FID: %.3f' % fid)
  return fid,return_image

#     with torch.no_grad():
#         for i in range(10):
#           label = torch.zeros((num, 10), device=DEVICE)
#           #n = random.randrange(0,9)
#           label[:, i] = 1
#           latent = torch.cat((z, label), dim=1)
#           output, = model.decoder(latent)
#           display_and_save_batch(f'{mode}-generation', output, f'-cifar10-{num}-{i}')

# model.eval()

# with torch.no_grad():
#     for batch_idx, (x, _) in enumerate(tqdm(test_loader)):
#         x = x.view(batch_size, x_dim)
#         x = x.to(DEVICE)
        
#         x_hat, _, _ = model(x)


#         break

# def show_image(x, y, idx):
#     x = x.view(batch_size,3, 32, 32)
#     y = y.view(batch_size,3, 32, 32)


#     plt.figure(figsize=(10,10))
#     plt.subplot(1,2,1)
#     plt.title('Label')
#     plt.axis('off')
#     plt.imshow(x[1].cpu().permute(1,2,0))
#     plt.subplot(1,2,2)
#     plt.title('Predict')
#     plt.axis('off')
#     plt.imshow(y[1].cpu().permute(1,2,0))
#     plt.show()

# show_image(x, x_hat, idx=0)

# #show_image(x_hat, idx=0)
# import torchvision

# def display_and_save_batch(title, batch, data, save=True, display=True):
#     """Display and save batch of image using plt"""
#     im = torchvision.utils.make_grid(batch, nrow=int(batch.shape[0]**0.5))
#     plt.title(title)
#     plt.imshow(np.transpose(im.cpu().numpy(), (1, 2, 0)))
#     if save:
#         plt.savefig('samples_cifar10/' + title + data + '.png', transparent=True, bbox_inches='tight')
#     if display:
#         plt.show()



# def generate_images(model=None, PATH=None, mode='random', num=16, grid_size=0.05):
#     """
#     Generates MNIST imgaes with 2D latent variables sampled uniformly with mean 0
#     Args:
#         mode: 'uniform' or 'random'
#         num: Number of samples to make. Accepts square numbers
#         grid_size: Distance between adjacent latent variables
#         PATH: The path to saved model (saved with torch.save(model, path))
#         model: The trained model itself
    
#     Note:
#         Specify only one of PATH or model, not both
#     """
#     if num!=(int(num**0.5))**2:
#         raise ValueError('Argument num should be a square number')
#     if PATH and model:
#         raise ValueError('Pass either PATH or model, but not both')
#     elif PATH is None and model is None:
#         raise ValueError('You passed neither PATH nor model')
    
#     # Load model
#     if PATH:
#         model = torch.load(PATH, map_location=DEVICE)
#     model.eval()

#     # Sample tensor of latent variables
#     if mode == 'uniform':
#         side = num**0.5
#         axis = (torch.arange(side) - side//2) * grid_size
#         x = axis.reshape(1, -1)
#         y = x.transpose(0, 1)
#         z = torch.stack(torch.broadcast_tensors(x, y), 2).reshape(-1, 2).to(DEVICE)
#     elif mode == 'random':
#         z = torch.randn((num, 128), device=DEVICE)
#     # Generate output from decoder
#     with torch.no_grad():
#         for i in range(10):
#           label = torch.zeros((num, 10), device=DEVICE)
#           #n = random.randrange(0,9)
#           label[:, i] = 1
#           latent = torch.cat((z, label), dim=1)
#           output, = model.decoder(latent)
#           display_and_save_batch(f'{mode}-generation', output, f'-cifar10-{num}-{i}')
