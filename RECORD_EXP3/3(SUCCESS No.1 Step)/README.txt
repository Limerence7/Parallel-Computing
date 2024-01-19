1. 参考编译命令：

gcc pivot.c -lm -o pivot

2. 参考运行命令：
（1）第一组
./pivot uniformvector-2dim-5h.txt
（2）第二组
./pivot uniformvector-4dim-1h.txt

注意：有两组测试数据，所以需要分别测试验证正确性

3. 正确性验证：
（1）第一组
diff result.txt refer-2dim-5h.txt
（2）第二组
diff result.txt refer-4dim-1h.txt