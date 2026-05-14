| 信息                                 | 内容                     | 原因                       | 处理办法                                                                                                                |
|------------------------------------|------------------------|--------------------------|---------------------------------------------------------------------------------------------------------------------|
| SYS_ALM404 ECC UNCORRECTABLE ERROR | CNC 的总线上 发生了问题。        | 可能是由于印刷板 的不良或外来噪声 的影响所致。 | 除了所显示的可能性最大的不良部位外，也有可能是主 板、系统报警画面上显示的 'MASTER PCB' 、 'SLAVE PCB' 的不良。 另外，该错误在某些情况下是由于外来噪声引起的。确 认机床附近是否有噪声源，是否已切实接地。 |
| SYS_ALM502 NOISE ON POWER SUPPLY   | 表示 CNC 的电 源中发生了噪声 或瞬断。 | 这是电源系统的异 常。              | 特定噪声等异常的原因并予以排除。也有可能是由于 SRAM 的数据被损坏所致。                                                                              |

## 10.24.4 系统报警 114 ～ 160 (FSSB 的报警 )

## 原因

FSSB 上检测出报警。

## 注释

报警信息中显示故障部位。表示部位的信息中的单词的含义如下所示。

MAIN :  CNC

内的伺服卡

AMPx :

表示从各线路的 CNC 数起第 x 台伺服放大器或者主轴放大器。

2 轴放大器、 3 轴放大器也作为 1 台计数。

SDUx :

从各线路的 CNC 数起第 x 台位置检测器接口单元

LINEx :

发生了报警的 FSSB 线路。

信息后面显示有 /LINEx 的情况下，表示主板（基本单元 A ）以及伺服卡（基本单元 G ）上的光连接器的编号。

LINE1 :

主板上的 COP10A 以及伺服卡上的 COP10A-1

| 信息                                                                                                                                                                                                                                                                                                                            | 内容·处理办法                                                                                                                                             |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------|
| SYS_ALM114 FSSB DISCONNECTION (MAIN -> AMP1) /LINEx SYS_ALM115 FSSB DISCONNECTION (MAIN -> SDU1) /LINEx SYS_ALM116 FSSB DISCONNECTION (AMPn -> AMPm) /LINEx SYS_ALM117 FSSB DISCONNECTION (AMPn -> SDU m) /LINEx SYS_ALM118 FSSB DISCONNECTION (SDU n -> AMP m) /LINEx SYS_ALM119 FSSB DISCONNECTION (SDU n -> SDU m) /LINEx  | ＜内容＞ 无法进行括弧内显示的单元间的 FSSB 通信。 ＜处理办法＞ 更换相应的主板、伺服卡、放大器、位置检 测器接口单元。 也有可能是由于相应的连接间的光缆所致。                                                                 |
| SYS_ALM120 FSSB DISCONNECTION (MAIN <- AMP1) /LINEx SYS_ALM121 FSSB DISCONNECTION (MAIN <- SDU 1) /LINEx SYS_ALM122 FSSB DISCONNECTION (AMPn <- AMPm) /LINEx SYS_ALM123 FSSB DISCONNECTION (AMPn <- SDU m) /LINEx SYS_ALM124 FSSB DISCONNECTION (SDU n <- AMP m) /LINEx SYS_ALM125 FSSB DISCONNECTION (SDU n <- SDU m) /LINEx | ＜内容＞ 无法进行括弧内显示的单元间的 FSSB 通信。 ＜处理办法＞ 更换相应的主板、伺服卡、放大器、位置检 测器接口单元。 也有可能是由于相应的连接间的光缆所致。 可能是由于括弧内右侧的单元的电源异常所 致。确认输入到单元的电源是否异常，单元 上连接的电机、编码器用的电缆内是否发生 短路。 |
| SYS_ALM126 SERVO AMP INTERNAL DISCONNECTION (AMPn) -> /LINEx SYS_ALM127 SERVO AMP INTERNAL DISCONNECTION (AMPn) <- /LINEx                                                                                                                                                                                                     | ＜内容＞ 在括弧内所显示的放大器内部检测出通信数 据的异常。 ＜处理办法＞ 更换相应的放大器。                                                                                                     |