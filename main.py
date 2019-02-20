import numpy as np 
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

img = mpimg.imread('wild.png')

#Now, let's look at the size of this numpy array object img as well as plot it using imshow.

print(img.shape)
plt.axis('off')
plt.imshow(img)


# In[65]:

#Okay, so the array has 711 rows each of pixel 996x3. Let's reshape it into a format that PCA can understand.
# 2988 = 996 * 3
img_r = np.reshape(img, (711, 2988))
print(img_r.shape)


# In[66]:

# Great, now lets run RandomizedPCA with 64 components (8x8 pixels) and transform the image.

ipca = PCA(64, svd_solver='randomized').fit(img_r)
img_c = ipca.transform(img_r)
print(img_c.shape)
print(np.sum(ipca.explained_variance_ratio_))

#Great, looks like with 64 components we can explain about 96% of the variance.


# In[69]:

#OK, now to visualize how PCA has performed this compression, let's inverse transform the PCA output and 
#reshape for visualization using imshow.
temp = ipca.inverse_transform(img_c)
print(temp.shape)

#reshaping 2988 back to the original 996 * 3
temp = np.reshape(temp, (711,996,3))

print(temp.shape)


# In[83]:

#Great, now lets visualize like before with imshow
plt.axis('off')
plt.imshow(temp.astype(np.uint8))
