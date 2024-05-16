#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <limits.h>
#include "mpi.h"

// 串行快速排序算法
int partition(int *arr, int low, int high);
void quicksort(int *arr, int low, int high);

int main(int argc, char *argv[])
{
    if (argc != 2)
    {
        printf("Please input the length of array!\n");
        exit(1);
    }
    int len = atoi(argv[1]);
    int *array = NULL;
    int i, j, k;
    double start2, end2;

    // 启动MPI环境
    int numprocs, myid;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);

    // 主进程随机生成长度为len的数列
    if (myid == 0)
    {
        array = (int *)calloc(len, sizeof(int));
        int *array_serial = NULL;
        array_serial = (int *)calloc(len, sizeof(int));
        srand(time(NULL));
        for (i = 0; i < len; i++)
        {
            array[i] = rand();
            array_serial[i] = array[i];
            // printf("array[%d] = %d\t", i, array[i]);
        }
        double start1, end1;
        start1 = MPI_Wtime();
        quicksort(array_serial, 0, len-1);
        end1 = MPI_Wtime();
        printf("Runtime(serial):   %lf\n", end1 - start1);
        free(array_serial);
    }

    // 开始计时
    MPI_Barrier(MPI_COMM_WORLD);
    if(myid == 0)
    {
        start2 = MPI_Wtime();
    }

    // (1) 均匀划分：len个元素均匀地划分成numprocs段，每台处理器有len/numprocs个元素
    int len_per_proc = len / numprocs;
    int *a = (int *)calloc(len_per_proc, sizeof(int));
    MPI_Scatter(array, len_per_proc, MPI_INT, a, len_per_proc, MPI_INT, 0, MPI_COMM_WORLD);
    /*
        MPI_Scatter()：将一段array的不同部分发送给所有的进程，
        array中的第一个元素发送给0号进程，第二个元素则发送给1号进程，以此类推。
        
        MPI_Scatter(
        void* send_data,            // 存储在0号进程的array
        int send_count,             // 具体需要给每个进程发送的数据的个数
        MPI_Datatype send_datatype, // 发送数据的类型
        void* recv_data,            // 接收缓存
        int recv_count,             // 缓存recv_count个数据
        MPI_Datatype recv_datatype, // 接收缓存的类型
        int root,                   // root进程的编号
        MPI_Comm communicator
        )
    */

    // (2) 局部排序：各处理器利用串行排序算法（此处采用串行快速排序），排序len_per_proc个元素
    quicksort(a, 0, len_per_proc-1);

    // (3) 正则采样：每台处理器各从自己地有序段中选取numprocs个样本元素
    int *samples = (int *)calloc(numprocs, sizeof(int));
    for (i = 0; i < numprocs; i++)
    {
        samples[i] = a[i * numprocs];
    }

    // (4) 样本排序：用一台处理器将所有numprocs*numprocs个样本元素用串行算法排序之
    int *samples_all;
    if (myid == 0)
    {
        samples_all = (int *)calloc((numprocs * numprocs), sizeof(int));
    }
    MPI_Gather(samples, numprocs, MPI_INT, samples_all, numprocs, MPI_INT, 0, MPI_COMM_WORLD);
    /*
        MPI_Gather()：从所有的进程中将每个进程的数据集中到根进程中，同样根据进程的编号对array元素排序
        
        MPI_Gather(
        void* send_data,
        int send_count,
        MPI_Datatype send_datatype,
        void* recv_data,
        int recv_count,  // 从单个进程接收的数据个数，不是总数
        MPI_Datatype recv_datatype,
        int root,
        MPI_Comm communicator
        )
    */
    if (myid == 0)
    {
        quicksort(samples_all, 0, (numprocs*numprocs) - 1);
    }

    // (5) 选择主元：用一台处理器选取numprocs-1个主元，并将其播送给其余处理器
    int *pivots = (int *)calloc((numprocs - 1), sizeof(int));
    if (myid == 0)
    {
        for (i = 0; i < (numprocs - 1); i++)
        {
            pivots[i] = samples_all[(i + 1) * numprocs];
        }
    }
    MPI_Bcast(pivots, (numprocs - 1), MPI_INT, 0, MPI_COMM_WORLD);
    /*
        MPI_Bcast()：
        当根进程调用的时候，void* data会被发送到其他所有的进程中
        当其他进程调用d的时候，void* data就会被根进程中的数据初始化
        
        MPI_Bcast(
        void* data,            // 数据
        int count,             // 数据个数
        MPI_Datatype datatype,
        int root,              // 根进程编号
        MPI_Comm communicator
        )
    */

    // (6) 主元划分：各处理器按主元将各自的有序段划分成numprocs个段
    int index = 0;
    int *partition_size = (int *)calloc(numprocs, sizeof(int)); // 记录每段的长度
    int *send_dis = (int *)calloc(numprocs, sizeof(int)); // 记录每段第一个元素的下标
    send_dis[0] = 0;
    for (i = 0; i < len_per_proc; i++)
    {
        if (a[i] > pivots[index])
        {
            index += 1;
            send_dis[index] = i;
        }
        if (index == (numprocs ))
        {
            partition_size[index-1] = len_per_proc - i+1;
            send_dis[index] = i;
            break;
        }
        partition_size[index]++;
    }

    // (7) 全局交换：各处理器将其辖段按段号交换到相应的处理器
    int *new_partition_size = (int *)calloc(numprocs, sizeof(int));
    MPI_Alltoall(partition_size, 1, MPI_INT, new_partition_size, 1, MPI_INT, MPI_COMM_WORLD);
    /*
        MPI_Alltoall()：每一个处理器给其他每一个处理器发送不同的数据
        例如：0号处理器收到所有处理器中的partition_size[0]
        
        MPI_Alltoall(
        void* send_data,
        int send_count,
        MPI_Datatype send_datatype,
        void* recv_data,
        int recv_data,
        MPI_Datatype recv_datatype,
        MPI_Comm communicator
        )
    */
    int totalsize = 0; // 每个处理器负责归并排序的数组总长度
    for (i = 0; i < numprocs; i++)
        totalsize += new_partition_size[i];
    int *new_partitions = (int *)calloc(totalsize, sizeof(int)); // 每个处理器负责归并排序的数组
    int *recv_dis = (int *)calloc(numprocs, sizeof(int)); // 记录新的每段第一个元素的下标
    int *end_dis = (int *)calloc(numprocs, sizeof(int)); // 记录新的每段最后一个元素的下标
    recv_dis[0] = 0;
    for (i = 1; i < numprocs; i++)
    {
        recv_dis[i] = recv_dis[i - 1] + new_partition_size[i - 1];
        end_dis[i - 1] = recv_dis[i]-1;
    }
    end_dis[numprocs-1] = totalsize-1;
    MPI_Alltoallv(a, partition_size, send_dis, MPI_INT, new_partitions, new_partition_size, recv_dis, MPI_INT, MPI_COMM_WORLD);
    /*
        MPI_Alltoallv()：往其他节点各发送（接收）多少数据，和MPI_Alltoallv()类似，参数不同
        例如：sendcounts[0]=3,代表要往0号处理器从sendbuf[sdispls[0]]开始发送3个数据
        
        MPI_Alltoallv(
        const void *sendbuf,
        const int *sendcounts,
        const int *sdispls, // 每个通信器进程的数据相对于 sendbuf 参数的位置。
        MPI_Datatype sendtype,
        void *recvbuf,
        const int *recvcounts,
        const int *rdispls， // 相对于每个通信器进程的数据的 recvbuf 参数的位置。
        MPI_Datatype recvtype,
        MPI_Comm communicator
        )
    */

    // (8) 归并排序：处理器使用归并排序将所接收的诸段施行排序
    int *sorted_partitions = (int *)calloc(totalsize, sizeof(int));
    for (i = 0; i < totalsize; i++)
    {
        int lowest = INT_MAX;
        int ind = -1;
        for (j = 0; j < numprocs; j++)
        {
            if ((recv_dis[j] <= end_dis[j]) && (new_partitions[recv_dis[j]] < lowest))
            {
                lowest = new_partitions[recv_dis[j]];
                ind = j;
            }
        }
        sorted_partitions[i] = lowest;
        recv_dis[ind] += 1;
    }

    // 主进程依次回收排好序的各段
    int *recv_count; // 主进程中记录每一段的长度
    if (myid == 0)
    {
        recv_count = (int *)calloc(numprocs, sizeof(int));
    }
    MPI_Gather(&totalsize, 1, MPI_INT, recv_count, 1, MPI_INT, 0, MPI_COMM_WORLD);
    /*
    MPI_Gather()：收集相同长度的数据块，本质上就是MPI_Scatter()的反向操作

        MPI_Gather(
        void* send_data,
        int send_count,
        MPI_Datatype send_datatype,
        void* recv_data,
        int recv_count,
        MPI_Datatype recv_datatype,
        int root,
        MPI_Comm communicator
        )
    */
    if (myid == 0)
    {
        recv_dis[0] = 0;
        for (i = 1; i < numprocs; i++)
            recv_dis[i] = recv_dis[i - 1] + recv_count[i - 1];
    }
    MPI_Gatherv(sorted_partitions, totalsize, MPI_INT, array, recv_count, recv_dis, MPI_INT, 0, MPI_COMM_WORLD);
    /*
        MPI_Gatherv()：与MPI_Gather()相似，不同的是可以从不同进程中接收不同数量的数据
        例如：进程1发送recvcounts[1]个数据给root进程
        
        MPI_Scatterv(
        const void *sendbuf,
        const int sendcounts[],
        const int displs[],
        MPI_Datatype sendtype,
        void *recvbuf,
        int recvcount,
        MPI_Datatype recvtype,
        int root,
        MPI_Comm communicator
        )
    */
    
    // 结束计时
    MPI_Barrier(MPI_COMM_WORLD);
    if (myid == 0)
    {
        end2 = MPI_Wtime();
        printf("Runtime(parallel): %lf\n", end2 - start2);
        // 输出排序后的结果
        // if (myid == 0)
        // {
        //     printf("\n排序结果\n");
        //     for (i = 0; i < len; i++)
        //     {
        //         printf("%d\n", array[i]);
        //     }
        // }
    }

    // 释放内存
    free(a);
    free(samples);
    free(pivots);
    free(partition_size);
    free(new_partition_size);
    free(new_partitions);
    free(sorted_partitions);
    free(send_dis);
    free(recv_dis);
    free(end_dis);
    if (myid == 0)
    {
        free(recv_count);
        free(array);
        free(samples_all);
    }

    // 结束MPI程序的运行
    MPI_Finalize();

    return 0;
}

void swap(int *a, int *b) 
{
	int temp;
	temp = *a;
	*a = *b;
	*b = temp;
}  

int partition(int *arr, int low, int high)
{
    int i = low;
    int j = high;
    int pivot = arr[(low + high) / 2]; // 取中间的数字作为基准值
    while (i <= j)
    {
        while (arr[i] < pivot)
            i++;
        while (arr[j] > pivot)
            j--;
        if (i <= j)
        {
            swap(arr+i, arr+j);
            i++;
            j--;
        }
    }
    return i;
}

void quicksort(int *arr, int low, int high)
{
    int new_low = partition(arr, low, high);
    int new_high = new_low - 1;
    if (low < new_high)
        quicksort(arr, low, new_high);
    if (new_low < high)
        quicksort(arr, new_low, high);
}