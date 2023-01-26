import abc
import picamera
import time
import numpy as np
import RPi.GPIO as GPIO
import time


import src
import matplotlib
from matplotlib import pyplot as plt

from src import SpotFinder
import matplotlib.pyplot as plt
import matplotlib.animation as animation


matplotlib.rcParams.update({'font.size':14})

def killfigure(k):
    if k == 1:
        ani.event_source.stop()

plt.ion()
plt.show()
while True:
    l = 60
    r = 250
    t = 90
    b = 250
    test = SpotFinder()
    pin = 27

    GPIO.setmode(GPIO.BCM)
    GPIO.setup(pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)

    print('\n')
    print('\n')
    print('\n')
    print('\n')
    print('\n')
    
    print('-------------------------------------------------------------')
    print('Pull the puck and start the experiment by pushing the blue button')


    GPIO.wait_for_edge(pin, GPIO.FALLING)
    print('Experiment started - release your puck in')
    print('4...')
    time.sleep(1)
    print('3...')
    time.sleep(1)
    print('2...')
    time.sleep(1)
    print('1...')
    time.sleep(1)
    print('Now!')

    test.start(sensor_mode=2,spot_size=5,aoi=(l,r,t,b),exposure=3000)
    
    print('Record finished - analyzing data...')

    t = np.array(test._t)

    t = (t - t[0])*1e-6

    x = np.array(test._x)
    y = np.array(test._y)


    w = []
    vx_s = []
    np.array(w)
    np.array(vx_s)

    vy_s = []
    np.array(vy_s)

    x = x-60
    y = y-90

    p00 = 45.65
    p10 = 0.01327
    p01 = -0.625
    p20 = 5.20e-5
    p11 = -5.617e-5
    p02 = 0.0001585


    p00_y =      -59.39
    p10_y =      0.6044
    p01_y =     0.01577
    p20_y =   3.851e-05
    p11_y =  -0.0002124
    p02_y =  -1.858e-06

    xtemp = x.copy()
    ytemp = y.copy()

    for i in range(0, x.shape[0]): #
        x[i] = p00 + p10 * xtemp[i] + p01 * ytemp[i] + p20 * xtemp[i]** 2 + p11 * xtemp[i] * ytemp[i] + p02 * ytemp[i]**2;
        y[i] = p00_y + p10_y * xtemp[i] + p01_y * ytemp[i] + p20_y * xtemp[i]** 2 + p11_y * xtemp[i] * ytemp[i] + p02_y * ytemp[i]**2;

    for i in range(1,x.shape[0]-1):
            dt = t[i+1] - t[i - 1]
            v_x = (x[i+1] - x[i - 1]) / dt
            v_y = (y[i+1] - y[i - 1]) / dt
            w_temp = (x[i] * v_y - y[i] * v_x) / (x[i] ** 2 + y[i] ** 2)
            w = np.append(w, w_temp)
            v_x = np.sqrt(v_x**2+v_y**2)
            v_y = np.sqrt(x[i]**2+y[i]**2)
            vx_s = np.append(vx_s,v_x)
            vy_s = np.append(vy_s,v_y)

    x = x*10/1000
    y = y*10/1000
    vx_s = vx_s*10/1000
    vy_s = vy_s*10/1000
    v_x = v_x*10/1000
    v_y = v_y*10/1000
    
    data = np.zeros([5,6])

    for i in range(0,5):
        temp1 = vy_s
        temp1 = temp1[30*i:30*i+29]
        temp1 = np.round(temp1,2)
        
        a_loc = np.argmax(temp1)
        p_loc = np.argmin(temp1)
        
        temp = vx_s
        temp = temp[30*i:30*i+29]
        temp = np.round(temp,2)
        temp2 = w
        temp2 = temp2[30*i:30*i+29]
        temp2 = np.round(temp2,2)
        data[i,0] = temp[a_loc]
        data[i,1] = temp[p_loc]
        data[i,2] = temp1[a_loc]
        data[i,3] = temp1[p_loc]
        data[i,4] = temp2[a_loc]
        data[i,5] = temp2[p_loc]
        
          
    x = np.delete(x, 0)
    y = np.delete(y, 0)
    t = np.delete(t, 0)

    x = list(x)
    y = list(y)
    t = list(t)
    w = list(w)
    vx_s = list(vx_s)
    vy_s = list(vy_s)

    w_min = np.min(w)
    w_max = np.max(w)
    
    v_min = np.min(vx_s)
    v_max = np.max(vx_s)
    v_diff = np.max(vx_s)-np.min(vx_s)
    
    #fig = plt.figure(figsize=(8,8), constrained_layout=True)
    #ax1 = fig.add_subplot(2, 2, 1)
    fig = plt.figure(figsize=(16,12))
    ax1 = plt.subplot2grid((3,3),(0,0), colspan=2,rowspan=2)
    ax2 = plt.subplot2grid((3,3),(0,2))
    ax3 = plt.subplot2grid((3,3),(1,2))
    ax4 = plt.subplot2grid((3,3),(2,0),colspan=3)
    
    ax1.set_aspect('equal')
    ax1.set_xlabel('x (m)')
    ax1.set_ylabel('y (m)')
    ax1.set_title('ORBIT')
    

    #ax2 = fig.add_subplot(2, 2, 2)
    ax2.set_ylim((w_min-3,w_max+3))
    #ax2.set_xlabel('time (s)')
    ax2.set_ylabel('$\omega$ (rad/s)')
    ax2.set_title('Angular Velocity')

    ax2.set_aspect('auto')

    #ax3 = fig.add_subplot(2, 2, 3)
    ax3.set_ylim((v_min-3*v_diff,v_max+3*v_diff))
    ax3.set_xlabel('time (s)')
    ax3.set_ylabel('$v$ (m/s)')
    ax3.set_aspect('auto')
    ax3.set_title('Velocity')
    #ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis('off')
    #ax4.axis('tight')
    
    columns = ('$v_{A} \ (m/s)$','$v_{P} \ (m/s)$','$r_{A} \ (m)$','$r_{P} \ (m)$','$\omega_{A} \ (rad/s)$','$\omega_{P} \ (rad/s)$')
    
    rows = ['1','2','3','4','5']
    
    im4 = ax4.table(cellText=data,
                    rowLabels=rows,
                    colLabels=columns,
                    loc='center')
    im4.scale(1,2.0)
    
    ims = []

    xdata = []
    ydata = []
    tdata = []
    wdata = []
    vxdata = []
    vydata = []
    

    for j in range(0, len(x) - 1,2):
        xdata.append(x[j])
        ydata.append(y[j])
        tdata.append(t[j])
        wdata.append(w[j])
        vxdata.append(vx_s[j])
        vydata.append(vy_s[j])

        im, = ax1.plot(xdata, ydata, 'b')
        im_0, = ax1.plot([0,x[j]], [0,y[j]], 'ro')

        im2, = ax2.plot(tdata, wdata,'b')
        #title = ax2.text(0.5, 1.05, "$\omega = $ {:.2f} rad/s".format(w[j]),
        #                size=plt.rcParams["axes.titlesize"],
        #                ha="center", transform=ax2.transAxes, )
        im3, = ax3.plot(tdata, vxdata,'b')
        #im4, = ax4.plot(tdata, vydata,'b')

        ims.append([im, im2,im3,im4,im_0])
    

    ani = animation.ArtistAnimation(fig, ims, interval=1, blit=True, repeat=False )

    
    plt.subplots_adjust(wspace=0.4,hspace=0.4)
    plt.draw()
    plt.pause(0.001)
    time.sleep(1)
    
    plt.suptitle('Push the button to finish this experiment')
    plt.draw()

    print('Push the button again to end this experiment and start a new one')

    GPIO.wait_for_edge(pin, GPIO.FALLING)
    #print('button detected')
    plt.close()
    
    #GPIO.wait_for_edge(pin, GPIO.FALLING)
    #ani._stop()
    #break


#ani.save('results.gif', writer='Pillow', fps=30)