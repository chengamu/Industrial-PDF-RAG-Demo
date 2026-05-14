## 5.4.2 数据批量保存

可以将 SRAM 数据和用户文件保存在输入输出设备中。

保存操作，通过 IPL 监控器的批量保存 / 恢复菜单进行。

## 注意

- 1 批量保存功能下保存到输入输出设备中的信息，全都只在相同的硬件配置、相同的选项配置的 CNC 上具有兼 容性。
- 2 利用本功能输出的文件的文件名已被固定。输入输出设备内已经有相同名称的文件时，该文件将被盖写。使用 本功能时，建议用户将输入 / 输出设备置于清零的状态。

## 批量保存操作说明

通过批量保存 / 恢复菜单的保存操作，将 SRAM 数据和用户文件保存到输入输出设备中。 有关将被保存的数据种类和文件名，请参阅本说明书的' 5.4.4 保存文件例'。

## 操作步骤

1. 将输入输出设备（存储卡、 USB 存储器）安装到 CNC 上。
2. 在 IPL 监控器上选择批量保存 / 恢复菜单。
3. 使用存储卡时，通过 MDI 键输入 "1" ，使用 USB 存储器，通过 MDI 键输入 "11" 。

## D4G1-0001

COPYRIGHT FANUC CORPORATION 2014

BATCH DATA BACKUP/RESTORE MENU

0. END
1. BATCH DATA BACKUP  (CNC -&gt; MEMORY CARD)
2. BATCH DATA RESTORE (MEMORY CARD -&gt; CNC)
11. BATCH DATA BACKUP  (CNC -&gt; USB MEMORY)
12. BATCH DATA RESTORE (USB MEMORY -&gt; CNC)

?

4. 显示确认信息，执行批量保存时，通过 MDI 键输入 "1" 。

输入 "0" 时，不会执行批量保存，再次显示批量保存 / 恢复菜单。

例 . 确认信息（存储卡）

BATCH DATA BACKUP (CNC -&gt; MEMORY CARD)

BATCH DATA BACKUP OK ? (NO=0,YES=1)

5. 批量保存中显示保存处理的进展状况。
6. 批量保存结束后，在文件大小旁显示信息 "END" ，再次显示批量保存 / 恢复菜单。通过 MDI 键输入 "0" ，结束批量保 存 / 恢复菜单。

BATCH DATA BACKUP OK ? (NO=0,YES=1) 1

SRAM DATA BACKUP : SRAM\_BAK.001 00240000/00240000 END

USER FILE BACKUP : PMC1.000     00020000/00040000

<!-- image -->

## 发生错误的情形

批量保存执行时发生错误时，画面上以红字显示信息，并再次显示批量保存 / 恢复菜单。有关错误的内容、对策，请参阅 本说明书的' 5.4.5 错误消息'。