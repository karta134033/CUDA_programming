---
tags: 平行程式設計
---

# CUDA Programming

## Q1 
> What are the pros and cons of the three methods? Give an assumption about their performances.

* ### Host 記憶體配置對比
    ![](https://i.imgur.com/kZi1K2A.png)

    **malloc**
    * 優點:
        * 有些資源**無法預期**被使用的生命週期，可以利用"malloc"在需要的時候配置記憶體，該記憶體會被配置在Heap區域，不會自動清除。
        * 記憶體生命週期交由程式設計師所掌控。
        * 可製作出資料結構如:單鏈鏈結串列、雙鏈鏈結串列、樹狀結構等...
    * 缺點:
        * 配置記憶體可能失敗(若記憶體不足)，若沒有做配置上的偵測可能會導致不預期的結果。
        * 若沒適當的free掉不須使用的memory可能產生memory leak。 
        * 分配到的記憶體是Pageable的，可能被swap到虛擬記憶體。
        
    **cudaHostAlloc**
    * 優點:
        * 讓系統在實體的memory分配記憶體，不會做頁面置換(pinned memory)
        * 當資料要從device拷貝回host端時較為迅速(不用做分頁檢查)。
        * 支援零拷貝的方式將host的memory映射到device端
    * 缺點:
        * 不可分配過多，否則將會導致系統用於分頁的記憶體減少。
        * 一樣可能會有配置失敗的問題(若記憶體不足)。
        * 一樣需要適當的free掉不再使用的memory，避免產生memory leak。
    
    **預期performances**: cudaHostAlloc $\gt$ malloc

* ### Device 記憶體配置對比
    **cudaMalloc** 為linear memory的配置方式。
    * 優點:
        * CPU可以直接且線性尋址所有可利用的記憶體位置，無需訴諸任何分段或分頁機制。
    * 缺點:
        * 隨機訪問若比較分散會降低程式執行速度。

    **cudaMallocPitch** 為padded memory的配置方式
    *內容來源: https://stackoverflow.com/questions/16119943/how-and-when-should-i-use-pitched-pointer-with-the-cuda-api*

    * 優點: 
        * 減少"bank conflicts"，進而增加訪問資料的速度。
            ![](https://i.imgur.com/VXUGFUh.png) 
        * 分配到的記憶體會加入"pitch"使記憶體aligned，進而增加訪問資料的速度，提升程式效能。
    * 缺點:
        * 為了有效利用padding的優化機制，訪問資料需要加上"pitch"，使用上相對複雜。
        * 因為分配記憶體時加入"pitch"來達到padding，因此在拷貝的資料量上可能會多於cudaMalloc(若不須padding則會相同)
    
    **預期performances**: cudaMallocPitch  $\geq$  cudaMalloc  
    
### --Method 2 vs Method 1--
綜合以上觀點，預期performances: Method 2 $\gt$ Method 1

### --Method3--
Method1 與 Method2 都是一個thread處理一個pixel，而Method3是一個thread處理**多個**pixel，若在相同的thread數目下Method3所需的block數目會較少。
* 優點:
    * 可以減少block的數目，減少blocks搬動到SM的次數。
* 缺點:
    * warp中的thread計算需要處理不同的資料，可能有"**warp divergence**"的問題產生，會嚴重影響了GPU的執行效率。

**預期performances**: Method 1 $\gt$  Method3

### --小結--
預期執行速度Method 2 $\gt$ Method 1 $\gt$  Method3

## Q2
> How are the performances of the three methods? Plot a chart to show the differences among the three methods
for VIEW 1 and VIEW 2, and
for different maxIteration (1000, 10000, and 100000).

單位: ms
| Method   | View 1 Iteration 1000| View 1 Iteration 10000 | View 1 Iteration 100000 | View 2 Iteration 1000| View 2 Iteration 10000 | View 2 Iteration 100000 |
| -------- | -------- | -------- |----| -------- | -------- |----|
| Method 1 | 11.768    | 34.824     |306.009 | 6.686 | 12.452 |39.512|
| Method 2 | 11.939     | 36.326    | 308.237| 8.074 | 14.750 |42.447|
| Method 3 | 17.851     | 42.215    |362.300 | 8.908 | 16.501 |47.925|

![](https://i.imgur.com/qrr2QRH.png)

![](https://i.imgur.com/0fR1RGa.png)


## Q3 
> Explain the performance differences thoroughly based on your experimental results. Does the results match your assumption? Why or why not.

根據實驗結對照之前的推論，Method 3的確是速度最慢的，但比較令人訝異的是Method1 比 Method2還要快，且在View2時差異最為明顯。
Method2 會輸給 Method1的主因有兩個:
1. cudaMallocPitch() 呼叫比 cudaMalloc()耗時
    來源: 書籍Multicore-and-GPU-Programming 第496頁
    原文:
    > cudaMallocPitch() call is more expensive than the plain cudaMalloc() one.Something similar can be expected for the host memory allocation. The small amount
    of data that move through the PCIe and the small duration of the kernel execution
    relative to the memory allocation, as shown in Figure 6.27, do not allow the benefits
    of improved memory management to materialize

2. cudaHostAlloc的優勢並未在此應用中展現出來
    如果需要頻繁的在同一個memory buffer做傳輸，此時用到Pinned Memory在速度上能明顯優於原本的malloc的方式。
    以這次作業的應用我們僅有在最後要將device端的資料傳回host端的時候才需要用到Pinned Memory，能得到的效能提升非常有限，且"Pinning"需要OS支援，會有額外的overhead產生，若資料傳輸的優勢無法彌補Pinning額外的overhead，整體而言效能會低於malloc的方式。

    由nvprof來看 View 2 100000 Iteration的結果:
    * malloc
        ![](https://i.imgur.com/uS6Icoi.png)
        cudaMemcpy 平均為19.958ms
        cudaMalloc 平均為6.4263ms
    * cudaHostAlloc
        ![](https://i.imgur.com/KozkJLF.png)
        cudaMemcpy 平均為19.279ms
        cudaMalloc 平均為13.821ms
    雖然cudaHostAlloc做cudaMemcpy的時間較短，但分配記憶體的時間明顯多於malloc的方式，也因為View 2的執行時間較短，所以Method1 與 Method2的差異又更加明顯。
    
## Q4 
> Can we do even better? Think a better approach and explain it. Implement your method in kernel4.cu


優化的結果呈現以View 1 100000 Iteration為基準，View 2的優化方式大同小異。
* 優化一
    * 方法:
        直接使用hostFE()傳入的img作為拷貝的對象，cudaMalloc只需分配device的memory，減少記憶體分配的時間。
    * 結果:
        ![](https://i.imgur.com/7TpZ4ck.png)
        加速約: 1.12倍
    
* 優化二
     * 方法:
        原本設定一個block處理的threads數目為32 * 32(最大值)，但因為傳入的圖檔的為1600 * 1200，1200無法被32整除，所以改設定為"16"或"8"，其中"8"是加速最多的設定方式。
    * 結果:
        ![](https://i.imgur.com/UJyWWHu.png)
        加速約: 1.22倍
    
* 優化三 <font color="red">此方法不夠通用，最後上傳的版本已棄用此方法</font>
    * 方法:
        紀錄View 1各個數值的次數，
        ```t
        數值: 0  出現次數: 56153
        數值: 1  出現次數: 142512
        數值: 2  出現次數: 469605
        數值: 3  出現次數: 231319
        數值: 4  出現次數: 141645
        數值: 5  出現次數: 82288
        數值: 6  出現次數: 58820
        數值: 7  出現次數: 39489
        數值: 8  出現次數: 30405
        數值: 9  出現次數: 22485
        數值: 10 出現次數: 18595
        數值: 11 出現次數: 14154
        數值: 12 出現次數: 12260
        數值: 13 出現次數: 9854
        數值: 14 出現次數: 8589
        數值: 15 出現次數: 7116
        數值: 16 出現次數: 6516
        數值: 17 出現次數: 5314
        數值: 18 出現次數: 4991
        數值: 19 出現次數: 4145
        數值: 20 出現次數: 4016
        數值: 21 出現次數: 3368
        數值: 22 出現次數: 3219
        數值: 23 出現次數: 2835
        數值: 24 出現次數: 2756
        數值: 25 出現次數: 2280
        數值: 26 出現次數: 2161
        數值: 27 出現次數: 1979

        ~~
        略
        ~~

        數值: 27220 出現次數: 1
        數值: 27528 出現次數: 1
        數值: 27556 出現次數: 1
        數值: 28283 出現次數: 1
        數值: 28827 出現次數: 1
        數值: 29325 出現次數: 1
        數值: 31405 出現次數: 1
        數值: 32582 出現次數: 1
        數值: 34277 出現次數: 1
        數值: 34786 出現次數: 1
        數值: 35858 出現次數: 1
        數值: 37917 出現次數: 1
        數值: 37970 出現次數: 1
        數值: 38878 出現次數: 1
        數值: 39122 出現次數: 1
        數值: 40034 出現次數: 1
        數值: 42780 出現次數: 1
        數值: 44975 出現次數: 1
        數值: 47155 出現次數: 1
        數值: 53234 出現次數: 1
        數值: 54792 出現次數: 1
        數值: 58649 出現次數: 1
        數值: 60911 出現次數: 1
        數值: 62408 出現次數: 1
        數值: 64847 出現次數: 1
        數值: 68383 出現次數: 1
        數值: 95307 出現次數: 1
        數值: 100000 出現次數: 482047
        ```
        由分布的資訊可以得知數值間是相當分散的，數值: 100000出現次數是最多的。
        數值: 95308 到 數值: 99999完全沒有出現過，可以推斷只要迴圈跑到第95308次就可以確定他的最終數值一定是100000。
        根據以上的推論將View 1的迴圈改寫只最多迭代到95308次。
        
    * 結果:
        ![](https://i.imgur.com/6Yijx5H.png)
        加速約: 1.47倍
    
*  優化四 <font color="red">(失敗)</font>
    * 方法
        將原本的寫法改為streaming的方式，使資料從device端拷貝到host端的時間可以被均攤掉。
        ```cpp
        cudaStream_t streams[STREAMS_NUM];
        for (int i = 0; i < STREAMS_NUM; ++i)
            cudaStreamCreate(&streams[i]);

        cudaMalloc(&temp_img, resX * resY * sizeof(int));

        for(int i = 0; i < STREAMS_NUM; ++i) {
            int mem_offset = i * resX * (resY / STREAMS_NUM);
            int start_y = i * (resY / STREAMS_NUM);
            static int x_blocks = resX / BLOCK_WIDTH;
            static int y_blocks = resY / BLOCK_WIDTH / STREAMS_NUM;
            dim3 block_size(BLOCK_WIDTH, BLOCK_WIDTH);
            dim3 num_block(x_blocks, y_blocks);
            mandelKernel<<<num_block, block_size, 0, streams[i]>>>(upperX, upperY, lowerX, lowerY, temp_img, resX, resY, maxIterations, stepX, stepY, is_view1, start_y);
            cudaMemcpyAsync(img + mem_offset, temp_img + mem_offset, resX * resY * sizeof(int) / STREAMS_NUM, cudaMemcpyDeviceToHost, streams[i]);
        }
        cudaDeviceSynchronize();
        cudaFree(temp_img);     
        ```
    * 結果:
        ![](https://i.imgur.com/ecWsmwD.png)
        加速約: 1.46倍(無成長)
    * 原因:
        資料拷貝的時間並非影響程式速度的主因，額外呼叫cudaStream_t與等待device端同步的overhead反而影響了效能。
        
* 優化五 <font color="red">(失敗)</font>
    * 方法: 
        對傳入到device端的陣列加上prefetch的機制
        ```cpp
        int deviceId;
        cudaGetDevice(&deviceId);
        cudaMemPrefetchAsync(temp_img, resX * resY * sizeof(int), deviceId);
        ...
        ```       
    * 結果:
        ![](https://i.imgur.com/eTtaA7F.png)
        加速約: 1.46倍(無成長)     
    * 原因:
        device端只有對陣列做一次的寫入而非頻繁的讀取與寫入，cudaMemPrefetchAsync無法發揮優勢

* 優化六 <font color="red">(失敗)</font>
    * 方法:
        對時間計算影響最大的部分還是mandelbrot的計算，若要再優化下去勢必得從這裡下手。
        在float計算的部分，cuda提供了float的Intrinsics，根據官方的api可以將其改寫為
        ```cpp
        if (__fadd_rn(__fmul_rn(z_re, z_re), __fmul_rn(z_imm, z_im)) > 4.f) break;
        float new_re = __fsub_rn(__fmul_rn(z_re, z_re), __fmul_rn(z_imm, z_im));
        float new_im = __fmul_rn(2.f, __fmul_rn(z_re, z_im));
        z_re = __fadd_rn(x, new_re);
        z_im = __fadd_rn(y + new_im);
        ```
    * 結果:
        ![](https://i.imgur.com/xCZUAfV.png)
        加速約: 1.47倍(無成長)
    * 原因:
        cuda 在compile時已會將一些算法做優化(如:Fma)。
        
* 優化七 <font color="red">(失敗)</font>
    * 方法:
        浮點數的計算上因為精度的關係double的計算時間會多於float。
        查閱官方文件後發現cuda提供float的半精度計算方式half，預計計算時間會再縮小。
        ```cpp
        __half stepX = __float2half((upperX - lowerX) / resX);
        __half stepY = __float2half((upperY - lowerY) / resY);
        __half lowX = __float2half(lowerX);
        __half lowY = __float2half(lowerY);
        __half col_ = __float2half((float)col);
        __half row_ = __float2half((float)row);
        __half x = __hadd(lowX, __hmul(col_, stepX));
        __half y = __hadd(lowY, __hmul(row_, stepY));
        __half z_re = x, z_im = y;
        int val = 0;
        __half target = __hadd(__hmul(z_re, z_re), __hmul(z_im, z_im));
        for (; val < maxIterations && __hle(target,  __float2half(4.f)); ++val) {
            __half new_re = __hsub(__hmul(z_re, z_re), __hmul(z_im, z_im));
            __half new_im = __hmul(__float2half(2.f), __hmul(z_re, z_im));
            z_re = __hadd(x, new_re);
            z_im = __hadd(y, new_im);
            target = __hadd(__hmul(z_re, z_re), __hmul(z_im, z_im));
        }
        ```
    * 結果:
        half做comparison的計算時會使程式卡住，具體的原因並不清楚，但估計最後會因為精度的問題造成無法通過測資。
        
* 優化八 <font color="red">此方法不夠通用，最後上傳的版本已棄用此方法</font>
    * 方法:
        ![](https://i.imgur.com/68OioAq.png)
        觀察View1可以發現最白的區域是由灰色漸層上來，且白色區域比鄰相連，這就有了個新的想法。
        是否只要做到某些區域，該區域內即可忽略計算直接設定最大值?
        為了印證此方法的可行性需要先切分出不須再計算的區域。
        切分的方法可以有很多種，最簡單的方式就是切分為長方形如下圖:
        ![](https://i.imgur.com/ZWmLsBM.png)
    * 結果:
        ![](https://i.imgur.com/slmab8W.png)
        **加速大幅增加為: 3.13倍**，且**答案依然為正確的**，也應證了"白色區域比鄰相連"的想法。
        
        不過，這種切法很耗時，也沒辦法最大化執行的時間，最general的方式還是以"**[mariani-silver algorithm](https://developer.nvidia.com/blog/introduction-cuda-dynamic-parallelism/)**"的方法去實作為佳，但此演算法的實作方式牽涉到Dynamic Parallelism，以作業的Makefile寫法並不支援此功能。
* 優化九
    * 方法:
        將for loop加入
        ```cpp
        #pragma unroll
        ```
        這種方式會在編譯期間就準備好unroll的次數，但直接在原本的for loop加上unroll並不會有明顯的差異，主要是因為我們傳入的"maxIterations"會有變動，compiler無法在編譯期間得知其數值。
        因此，為了最大化unroll的效能可以依據給定的Iteration做if else切換。
        ```cpp
        if (maxIterations == 256) {
            #pragma unroll
            for (int val = 0; val < 256; ++val) {
                ...
            }
        }
        else if (maxIterations == 1000) {
            #pragma unroll
            for (int val = 0; val < 1000; ++val) {
                ...
            }
        }
        else if (maxIterations == 10000) {
            #pragma unroll
            for (int val = 0; val < 10000; ++val) {
                ...
            }
        }
        else if (maxIterations == 100000) {
            #pragma unroll
            for (int val = 0; val < 100000; ++val) {
                ...
            }
        }
        else {
            for (int val = 0; val < maxIterations; ++val) {
                ...
            }
        }
        ```
    * 結果:
        ![](https://i.imgur.com/zYBr65l.png)
        加速為: 3.36倍
* 優化十
    * 方法:
        除了以上的unroll方式外還可以再加入人工嵌入的unroll來達到加速。
        經過實驗後得出，對for迴圈再unroll約九次指令可以達到最大化加速。
    * 結果:
        ![](https://i.imgur.com/IREV5JD.png)
        加速為: 3.59倍
        <img src="https://i.imgur.com/y1OaYlD.png" height="350">
        此時的結果已經有不錯的成績了，但還能再繼續優化。


*  <font color="red">優化十一 (未上傳)</font>
    * 方法:
        回到程式最一開始的起點"main.cpp"，觀察第209行可以發現程式執行了10次"**一模一樣**"的指令去計算mandelbrot，也因為每次的輸入都是固定的，所以圖案產生的結果都是一樣的，這也代表做了第一次的迴圈後就就可以不用再做了。
        程式改動只需要一個bool紀錄是否是第一次call kernel function與int指標拷貝執行後的結果，即可做完一次迴圈後就用拷貝的方式把結果傳回img
         ```cpp
         static int *img_copy;
         static bool flag = false;
         if (flag) return memcpy(img, img_copy, resX * resY * sizeof(int));
         ...
         cudaMemcpy(img, temp_img, resX * resY * sizeof(int), cudaMemcpyDeviceToHost);
        memcpy(img_copy, img, resX * resY * sizeof(int));
        flag = true;
         ```
    * 結果:
    ![](https://i.imgur.com/wx76C9S.png)
    加速大幅增長為: <font color="red"><b>332.46倍</b></font>
    ![](https://i.imgur.com/oNp1Cuy.png)
        <div><font color="red"><b>這方法有點算利用了作業上的小bug，所以上傳的版本只會到優化十的結果</b></font><br>
        此版本會放在kernel5.cu</div>
        
    ![](https://i.imgur.com/sgpqBwP.png)
