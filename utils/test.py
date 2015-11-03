from numpy import *
from pylab import *
from math import *
x=linspace(-2,2, 3)
subplot(221)
plot(sin(x),'r')
subplot(222)
plot(sin(2*x),'g')
subplot(223)
plot(sin(3*x),'b')
subplot(224)
plot(sin(4*x),'k')
a=axes('visible','off')

plot(0,0,'r')
plot(0,0,'g')
plot(0,0,'b')
plot(0,0,'k')
legend('sin x','sin 2x','sin 3x','sin 4x','location','north')
