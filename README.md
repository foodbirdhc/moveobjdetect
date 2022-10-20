# moveobjdetect
use for moving target detection, especially for night kitchen mouse detection


基于opencv的超简单的移动物体跟踪和检测，目前只适用于背景稳定不变的场景，比如比较适用在夜间针对厨房进行老鼠检测

亮点：
        重新实现了opencv的耗时接口，在pc上运行时间为1ms左右，非常轻量


编译方法：

1.首先解压opencv依赖库
>> tar zxvf third_party.tar.gz

2.编译依赖库
>> cd src
>> mkdir build; cd build; cmake ..; make install

3.编译sample
>> mkdir build; cd build; cmake ..; make install
