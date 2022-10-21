# moveobjdetect
use for moving target detection, especially for night kitchen mouse detection


基于opencv的超简单的移动物体跟踪和检测，目前只适用于背景稳定不变的场景，本次项目一开始的目的也是用在夜间厨房的老鼠检测，目前性能和精度都比较不错

说明：
        重新实现了opencv的耗时接口，在pc上运行时间为1ms左右，非常轻量, 支持跨平台


编译方法：

编译器版本：gcc 5.4.0 (或者其他版本编译时，需要重新编译对应的opencv）

1.首先解压opencv依赖库
>> tar zxvf third_party.tar.gz

2.编译依赖库
>> cd src
>> mkdir build; cd build; cmake ..; make install

3.编译sample
>> mkdir build; cd build; cmake ..; make install
