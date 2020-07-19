"""
One-layer spiking neural network with LIF neurons for classification of the "digits" dataset from sklearn
Author: Ali Safa
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn.datasets import load_digits
from sklearn.preprocessing import OneHotEncoder

def sample_spherical(npoints, ndim): #sample a vector of dimension "ndim" from the unit sphere randomly
    vec = np.random.randn(ndim, npoints)
    vec /= np.linalg.norm(vec, axis=0)
    return vec[:,0]

def normalize_imges(data): #normalize pixel values to [-1, 1] 
    for i in range(data.shape[0]):
        img = data[i]
        data[i] = 2*(img - min(img))/(max(img) - min(img)) - 1
    return data

def simulate_neuron(Tsim, dt, trc, tref, vrest, vth, J):
    N = int(np.round(Tsim/dt))
    Vprev = 0
    Jprev = 0
    spike_train = np.zeros(N)
    Vhist = np.zeros(N)
    mutex = 0  
    
    for i in range(N):
        if mutex == 0:
            V = (J[i] + Jprev - (1-2*trc/dt)*Vprev)/(1+2*trc/dt) #bilinear transform
            if V < vrest:
                V = vrest
            elif V > vth:
                spike_train[i] = 1
                V = vrest
                mutex = np.round(tref/dt)
            Vhist[i] = V 
            Jprev = J[i]
            Vprev = V
        else:
            mutex -= 1
    
    return Vhist, spike_train

def Gaussian_filter(Tsim, dt, tau):
    t = np.linspace(0,Tsim,int(np.round(Tsim/dt)))
    h = np.exp(-((t-Tsim/2)**2)/tau**2)
    h = (1/dt)*h/np.sum(h)
    return h

def PSC_filter(Tsim, dt, tau):
    t = np.linspace(0,Tsim,int(np.round(Tsim/dt)))
    h = np.exp(-(t-Tsim/2)/tau)
    h[0:len(h)//2] = 0
    h = (1/dt)*h/np.sum(h)
    return h
    
np.random.seed(18945) #to get reproducable results
plt.close('all')
digits = load_digits()

D = 64 #data dimensions
M = 256 #number of neuron in population 256
train_ratio = 0.9
N_train = int(train_ratio*digits.data.shape[0]) #nbr of examples
F_max_l = 100 #100
F_max_h = 200
in_l = -1.0
in_h = 1.0
tref = 0.002 #2ms
trc = 0.02 #20ms

x = normalize_imges(digits.data)
x_train = x[:int(train_ratio*x.shape[0]),:] #spliting in training and test sets
x_test = x[int(train_ratio*x.shape[0]):,:]
labels = digits.target.reshape(-1,1)
enc = OneHotEncoder(handle_unknown='ignore')
enc.fit(labels)
one_hot = enc.transform(labels).toarray()
one_hot_train = one_hot[:int(train_ratio*x.shape[0]),:]
one_hot_test = one_hot[int(train_ratio*x.shape[0]):,:]

"""
Training step
"""
A_train = np.zeros((N_train, M))
J_bias_vec = np.zeros(M)
e_vec = np.zeros((M,D))
alpha_vec = np.zeros(M)

for i in range(M): #generate neurons randomly, record output and save the model
    amax = np.random.uniform(F_max_l,F_max_h,1) # maximum rate uniformly distributed between 100 and 200 HZ
    #xi = np.random.uniform(in_l+0.05,in_h-0.05,1) # x-intercept
    xi = np.random.uniform(-0.05,in_h,1) # new idea x-intercept
    alpha = (1/(1-np.exp((tref - 1/amax)/trc)) - 1)/(1-xi) #for LIF neuron
    alpha_vec[i] = alpha
    Jbias = 1-xi*alpha
    J_bias_vec[i] = Jbias
    e = sample_spherical(1, D)
    e_vec[i,:] = e
    a_x = np.zeros(N_train)
    #for LIF neuron -> should be vecorized ideally
    for j in range(N_train):        
        if np.multiply(alpha, np.inner(e, x_train[j,:])) + Jbias > 1: 
            a_x[j] = 1/(tref - trc*np.log(1 - 1/(np.multiply(alpha, np.inner(e, x_train[j,:])) + Jbias)))
        else:
            a_x[j] = 0
        
    A_train[:,i] = a_x


mu, sigma = 0, 0.01*np.max(A_train) # mean and standard deviation
s = np.random.normal(mu, sigma, A_train.shape)
A_noisy = np.add(A_train, s) 

#More stable to use lstq instead of inv
d_reg = np.linalg.lstsq(
A_train.T @ A_train + 0.5 * N_train * np.square(sigma) * np.eye(M), A_train.T @ one_hot_train,
rcond=None)[0].T


x_hat = np.dot(A_noisy, d_reg.T)
MSE = np.mean(np.power(one_hot_train-x_hat, 2))

"""
Testing step
"""
N_test = digits.data.shape[0] - int(train_ratio*digits.data.shape[0]) #nbr of examples

Tsim = 0.1
dt = 0.001
vrest = 0
vth = 1
Tlen = int(np.round(Tsim/dt))
A_test = np.zeros((N_test, Tlen, M))
S_test = np.zeros((N_test, Tlen, M))
#h = Gaussian_filter(Tsim, dt, 0.03) #non-causal
h = PSC_filter(Tsim, dt, 0.05) #causal
t = np.linspace(0, Tsim, Tlen)

for i in range(M): #pops saved parameters of neurons and record output
    alpha = alpha_vec[i]
    Jbias = J_bias_vec[i]
    e = e_vec[i,:]
    a_x = np.zeros((N_test, Tlen))
    s_x = np.zeros((N_test, Tlen))
    #for LIF neuron -> should be vecorized ideally
    for j in range(N_test):        
        J = np.multiply(alpha, np.inner(e, x_test[j,:])) + Jbias
        Jin = np.concatenate((Jbias*np.ones(Tlen//4), J*np.ones(3*Tlen//4))) #present input to network at Tsim//2
        #Jin = J*np.ones(Tlen)
        Vhist, spike_train = simulate_neuron(Tsim, dt, trc, tref, vrest, vth, Jin)  
        a_x[j,:] = np.convolve(spike_train, h, 'same')
        spike_train[spike_train == 0] = -1
        s_x[j,:] = np.multiply(spike_train, t)
        
    A_test[:,:,i] = a_x
    S_test[:,:,i] = s_x

error_history = np.zeros(Tlen)
for t in range(Tlen):
    x_hat_test = np.dot(A_test[:,t,:], d_reg.T) #decode activity of the population
    
    cnt = 0
    for i in range(N_test): #count error rate of model    
        if np.argmax(x_hat_test[i]) == np.argmax(one_hot_test[i]):
            cnt += 0
        else:
            cnt += 1
    
    error_rate = 100 - 100*cnt/N_test #compute error rate of model
    error_history[t] = error_rate
    #print('Accuracy of model: ', round(error_rate, 2), '%')

t = np.linspace(0, Tsim, Tlen)
plt.figure(1)    
plt.plot(t, error_history, '.-')
plt.title('Accuracy of model in function of time')
plt.xlabel('Time [s]')
plt.ylabel('Accuracy on test set [%]')
plt.grid('on')
plt.show()

#Accuracy evolution over time
Y = error_history

fig, ax = plt.subplots(1,1)
ax.set_xlim([0, Tsim])
ax.set_ylim([-1, 110])

accuracygraph, = ax.plot([], [])
dot, = ax.plot([], [], 'o', color='red')
props = dict(boxstyle='square', facecolor='white', alpha=0.5)
textstr = "Accuracy: " + str(np.trunc(Y[0]*10)/10) + "%" 
box = ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=18,
        verticalalignment='top', bbox=props)
def accuracy_anim(i):
    accuracygraph.set_data(t[:i],Y[:i])
    dot.set_data(t[i],Y[i])
    textstr = "Accuracy: " + str(np.trunc(Y[i]*10)/10) + "%" 
    box.set_text(textstr)

anim = animation.FuncAnimation(fig, accuracy_anim, frames=len(t), interval=1)
plt.title('Accuracy of model in function of time')
plt.xlabel('Time [s]')
plt.ylabel('Accuracy on test set [%]')
plt.grid('on')
plt.show()

Writer = animation.writers['ffmpeg']
writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
anim.save('im.mp4', writer=writer) 

#Neural activity
events = S_test[0,:,:].T
fig1, ax1 = plt.subplots(1,1)
eventgraph = ax1.eventplot(events, colors='red')
def event_anim(i):
    ax1.set_xlim([0.001, i*0.001+0.001])

anim1 = animation.FuncAnimation(fig1, event_anim, frames=len(t), interval=1)
plt.grid('on')
plt.title('Neural Activity')
plt.xlabel('Time [s]')
plt.ylabel('Neuron index')
plt.show()

anim1.save('im1.mp4', writer=writer)
