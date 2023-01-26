    #!/usr/bin/env python3

import abc
import picamera
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# --- Video frame analysis utility functions ----------------------------------

def edgemedian(data, border):

    """Find the median of the pixels in a band at the edge of the data

    Args:
        data (numpy array): rectangle of pixels
        border (int): width of border to find median within

    Returns:
        float: The median pixel value

    Examples:
        >>> x = np.array([[1,2,3,1],[3,9,9,2],[2,9,9,3],[1,3,2,1]])
        >>> edgemedian(x,1)
        2.0
        >>> edgemedian(x,2)
        2.5
    """

    edge = np.ones(shape=data.shape, dtype=bool)    # Array of True
    edge[border:-border, border:-border] = 0        # Set middle False
    return np.median(data[edge])                    # Median of edge


def getbox(pos, size, width):

    """Get a slice surrounding a pixel, clipped as necessary

    Note:
        A coordinate which is an integer refers to the centre of a pixel,
        equivalently, coordinates in the range -0.5 to +0.5 will be regarded as
        within the pixel at index 0.

    Args:
        pos (number): centre coordinate
        size (int): number of pixels each side of pixel containing the centre
        width (int): slice then moved if necessary to fit within [0,width)

    Returns:
        (tuple): bounds of the slice

    Examples:
        >>> getbox(4, 5, 100)
        (0, 11)
        >>> getbox(5.49, 5, 100)
        (0, 11)
        >>> getbox(5.51, 5, 100)
        (1, 12)
        >>> getbox(93.49, 5, 100)
        (88, 99)
        >>> getbox(93.51, 5, 100)
        (89, 100)
        >>> getbox(95, 5, 100)
        (89, 100)
    """

    low = int(round(pos)) - size
    high = int(round(pos)) + size + 1

    if low < 0:
        high -= low
        low = 0

    if high > width:
        low -= (high - width)
        high = width

    return low, high


def locate_spot(image):

    """Find the coordinates of the pixel with the largest value

    Args:
        image: A 2D numpy array of pixel values

    Returns:
        A tuple (x, y) of the position of the spot
    """
    
    y, x = np.unravel_index(image.argmax(), image.shape)
    return x, y


def refine_spot(image, spot_size, x, y):

    """Refine an estimate of the position of the spot

    Args:
        image: A 2D numpy array of pixel values
        spot_size: The size of the spot to look for
        x, y: An initial estimate of the position of the spot

    Returns:
        A tuple (x, y) of the refined position of the spot

    Notes:
        This function subtracts an estimate of the background and then computes
        the centre of gravity of the pixel values surrounding the crude
        estimate of the spot position. This process is iterated a couple of
        times.
    """

    nit = 2
    border = 2

    h, w = image.shape

    for it in range(nit):

        # Get a small box surrounding the spot

        lox, hix = getbox(x, spot_size, w)
        loy, hiy = getbox(y, spot_size, h)

        n = image[loy:hiy, lox:hix]

        # Calculate the centre of gravity of the data within that box,
        # first subtracting the background level (determined as the median
        # value of the pixels in some border at the edge of the box).

        m = edgemedian(n, border)
        n = n - m

        f = np.sum(n)
        if f == 0:          # Whole box the same intensity...
            break           # ...give up now

        j, i = np.indices(n.shape)

        x = np.sum(i*n) / f + lox
        y = np.sum(j*n) / f + loy

    # Return results

    return x, y


# --- Spot finder class -------------------------------------------------------

class SpotFinder(abc.ABC):

    """Class to use a Raspberry Pi camera to find the location of a spot

        To use this class, define your own subclass overriding at least the
        spot_handler() method, create an instance of that class, and call the
        start() method.

        You may also wish to override the find_spot() method to change the spot
        finding algorithm. Your override function may, but need not, use either
        or both of the utility functions locate_spot() and refine_spot().
    """

    def start(
        self,
        *,
        spot_size,
        sensor_mode=1,
        resolution="320x304",
        framerate=30,
        exposure=5000,
        duration=10,
        aoi=None
    ):

        """Start spot location processing

        Args:
            spot_size: the size of the spot, in pixels

            sensor_mode: the sensor mode to use for the camera [*]

            resolution: the resolution of the returned video frames [*]

            framerate: the framerate, in frames per second, of the video [*]

            exposure: the exposure time, in microseconds [*]

            duration: the duration, in seconds, to process video

            aoi: a tuple (left, right, top, bottom) of a subregion of the frame
                within which the spot is known to be located. The default value
                of None means the whole frame is searched.

        Notes:
            Raspberry Pi camera documentation, for arguments marked [*], can be
            found at https://picamera.readthedocs.io
        """

        self._spot_size = spot_size
        self._aoi = aoi
        self._x = []
        self._y = []
        self._t = []
    
        with picamera.PiCamera(
            clock_mode='raw',
            sensor_mode=sensor_mode,
            resolution=resolution,
            framerate=framerate
        ) as self._camera:
            self._camera.shutter_speed = exposure
            time.sleep(3)                       # let the camera warm up

            self._camera.start_recording(self, format='rgb')
            self._camera.wait_recording(duration)
            self._camera.stop_recording()

        self._camera = None

    def write(self, buf):

        """Handle a video frame"""

        # Extract timestamp and create np array view onto the passed frame data

        ts = self._camera.frame.timestamp
        width, height = self._camera.resolution

        image = np.frombuffer(buf, dtype=np.uint8, count=width*height*3)
        image = image.reshape((height, width,3))
        
        image = image[:,:,0]
        #image = image[80:250,50:250]
        
        image = image.copy()
        
        image[160:180,150:170] = 0
        
        #print(image.shape)

        # Locate the spot, but if an area of interest is specified, only look
        # there
        
        xtemp = [1]
        ytemp = [1]

        if self._aoi is not None:
            image = image[self._aoi[2]:self._aoi[3], self._aoi[0]:self._aoi[1]]

        image[:,180:-1] = 0

        x, y = self.find_spot(image, self._spot_size)
        

        if self._aoi is not None:
            x += self._aoi[0]
            y += self._aoi[2]
     
     
               
        self._x.append(x)
        self._y.append(y)
        self._t.append(ts)
        #self.spot_handler(ts, x, y)

    def find_spot(self, image, spot_size):
        """Find the spot in an image

        Args:
            image:
            spot_size:

        Returns:

        Notes:
            The default implementation calls locate_spot to find an approximate
            location of the spot, and then refine_spot to improve that
            estimate.

            This function may be overridden in a subclass if a different spot
            location algorithm is wanted.
        """
        #import matplotlib.pyplot as plt
        #plt.imshow(image)
        
        #plt.show()
        #import numpy
        #numpy.savetxt('image.csv',image, delimiter=",")
        x, y = locate_spot(image)
    
        

        x, y = refine_spot(image, spot_size, x, y)
        return x, y

    #@abc.abstractmethod
    #def spot_handler(self, timestamp, x, y):
        '''Take any desired action for a located spot in a frame

        Args:
            timestamp: timestamp in microseconds from an arbitrary epoch
            x, y: pixel coordinates (floats) of the spot

        Notes:
            This function must be overridden in a subclass.
        '''
        #pass

    def flush(self):
        """Called when all video frames have been processed

        Notes:
            This function may be overridden in a subclass if any action is
            needed when all video frames have been processed.
        """
        pass
