#include <stdio.h>
#include <stdlib.h>
// #include <cutil_inline.h>
#include <helper_functions.h>



//宏及全局变量定义
//////////////////////////////////////////////////////////////////////////
#define RowSize 5 //定义5*5的lattice路网
#define NodeNumTol 25
// Do not consider the road beyond the lattice
#define RoadNum 40
#define NodeConnectNumMax 4

#define NumofRoads 4
// lanes for a road
#define NumofLanes 3

#define RoadSIZE 130 //行驶道路车辆最大数目
// in meter
#define VehicleLength 4

#define pi 3.1415926


#define NumofPhase 4  //每个路口有4个信号灯控制相
// min for phase time
#define Cmin 120
// max for phase time
#define Cmax 240
// min green time
#define GreenTimeInPhaseMin 6
// max green time
#define GreenTimeInPhaseMax 90

#define flow 0.01

#define MaxGen 1000
#define PeopleNum 200

#define PCross 0.95
#define PMutate 0.05
/************************************************************************/
/* Init CUDA                                                            */
/************************************************************************/

struct VehicleAgent
{
	//车辆序号
	// smaller number is in front
	int VehicleID;

	float position;//位置
	float speed;//速度
	float acceleration;//加速度
	bool VehicleFlag;//标示本单元格内是否有车
	float ExpSpeed;//期望速度

	//车辆在单个路段行驶耗时
	int RoadTimeConsumed;
};

struct RoadAgent
{
	//道路队列的定义
	VehicleAgent BufferRoad[RoadSIZE];//队列主体

	int RoadFront;//头指针
	int RoadRear; //尾指针
	int RoadCount;//队列中的元素个数

};

struct TrafficLightsAgent
{
	//四项的绿灯状态
	// west and east go, then left, then south and north go, then left
	bool GreenLightStatus_Phase1;
	bool GreenLightStatus_Phase2;
	bool GreenLightStatus_Phase3;
	bool GreenLightStatus_Phase4;

	int cycle;//信号灯周期

	int Phase;//与基准信号灯的相位差

	//四相的绿灯持续时间
	int GreenDuration__Phase1;
	int GreenDuration__Phase2;
	int GreenDuration__Phase3;
	int GreenDuration__Phase4;	
};

//定义描述路网拓扑的数据结构
struct RoadNetworkResTemplate
{
	int NodeIndex;

	//定义本节点的四条道路与其他节点相连的节点序号
	// 0, north, 1 west, 2 south, 3 east
	int NodeLinkTo[NumofRoads];
	// 0, south, 1 east, 2 north, 3 west
	int RoadLinkTo[NumofRoads];

	//定义交通灯
	TrafficLightsAgent TrafficLight;


	//定义用于计算路段关联度的变量//////////////////////////////////
	//所有数据均为流入车辆数据
	float VehicleArrRate[NumofRoads];

	int RoadLength[NumofRoads];//通向本节点道路的路长	

	float RoadTraveTime[NumofRoads];//通向本节点道路上的车辆总行驶时间
	
	int  VehicleInNumInAll[NumofRoads];//流向本节点的车辆总数
	int RoadVehiclePassNum[NumofRoads][NumofLanes];//路段通过的车辆数目
	int  VehicleInNumInMax[NumofRoads];//流向本节点的最大车辆数

	//存储路网关联度数据
	float RoadRelationNum[NumofRoads];
	////////////////////////////////////////////////////////////////

	//创建单路口连接的4条道路
	RoadAgent Road[NumofRoads][NumofLanes];

};

RoadNetworkResTemplate RoadNetworkRes[NodeNumTol];
RoadNetworkResTemplate RoadNetworkResNew[NodeNumTol];


int SimulationClock;//定义仿真时钟(s)
int SimulationDuration;//定义仿真持续时间(s)

//各节点间关联度矩阵
float LapMatirx[NodeNumTol][NodeNumTol];
float LapMatirx2[NodeNumTol][NodeNumTol];
float LapVector[RoadNum];

//画图用的数据矩阵
float Paint[NodeNumTol];

//遗传算法用到的染色体矩阵
struct People
{
	int PeopleIndex;
	int Chomosome[RoadNum];
	int LapMatrixBinary[NodeNumTol][NodeNumTol];

	//子区数目
	int N;
	//所有子区的区域关联度--最多划分为25个子区 即一个路口一个子区 完全独立
	float NRelation[NodeNumTol][NodeNumTol];

	float fitness;
};
People PeopleCurrent[PeopleNum],PeopleNew[PeopleNum];


struct Tree
{
	int NodeIndex;
	int Leaf[4];
};

Tree tree[NodeNumTol];


bool NodeFlag[NodeNumTol];
bool LeafFlag[NodeNumTol];
int temp[NodeNumTol];
float temp2[NodeNumTol];
float temp3[NodeNumTol];

/************************************************************************/
/* Init CUDA                                                            */
/************************************************************************/
#if __DEVICE_EMULATION__

bool InitCUDA(void){return true;}

#else
bool InitCUDA(void)
{
	int count = 0;
	int i = 0;

	cudaGetDeviceCount(&count);
	if(count == 0) {
		fprintf(stderr, "There is no device.\n");
		return false;
	}

	for(i = 0; i < count; i++) {
		cudaDeviceProp prop;
		if(cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
			if(prop.major >= 1) {
				break;
			}
		}
	}
	if(i == count) {
		fprintf(stderr, "There is no device supporting CUDA.\n");
		return false;
	}
	cudaSetDevice(i);

	printf("CUDA initialized.\n");
	return true;
}

#endif
/************************************************************************/
/* Example                                                              */
/************************************************************************/
__global__ static void HelloCUDA(char* result, int num)
{
	
}



void RoadNetworkInitial()
//定义路网拓扑 明确连接关系
{	
	for (int i=0;i<NodeNumTol;i++)
	{
		
		RoadNetworkRes[i].NodeIndex=i;

		for (int k=0;k<NumofRoads;k++)
		{
			RoadNetworkRes[i].NodeLinkTo[k]=-1;	
			RoadNetworkRes[i].RoadLinkTo[k]=-1;	
		}		
	}

	for (int i=0;i<NodeNumTol;i++)
	{
		if (i%RowSize==0)
			//对应矩阵第一列元素
		{	
			if (i==0)
			{
				// starts from north 0, then west 1, then south 2, then east 3
				RoadNetworkRes[i].NodeLinkTo[0]=-1;
				// starts from south 0, then east 1, then north 2, then west 3
				RoadNetworkRes[i].RoadLinkTo[0]=-1;

				RoadNetworkRes[i].NodeLinkTo[1]=-1;
				RoadNetworkRes[i].RoadLinkTo[1]=-1;

				RoadNetworkRes[i].NodeLinkTo[2]=i+RowSize;
				RoadNetworkRes[i].RoadLinkTo[2]=0;

				RoadNetworkRes[i].NodeLinkTo[3]=i+1;
				RoadNetworkRes[i].RoadLinkTo[3]=1;

			}
			else if (i==RowSize*(RowSize-1))
			{
				RoadNetworkRes[i].NodeLinkTo[0]=i-RowSize;
				RoadNetworkRes[i].RoadLinkTo[0]=2;

				RoadNetworkRes[i].NodeLinkTo[1]=-1;
				RoadNetworkRes[i].RoadLinkTo[1]=-1;

				RoadNetworkRes[i].NodeLinkTo[2]=-1;
				RoadNetworkRes[i].RoadLinkTo[2]=-1;

				RoadNetworkRes[i].NodeLinkTo[3]=i+1;
				RoadNetworkRes[i].RoadLinkTo[3]=1;
			}
			else
			{
				RoadNetworkRes[i].NodeLinkTo[0]=i-RowSize;
				RoadNetworkRes[i].RoadLinkTo[0]=2;

				RoadNetworkRes[i].NodeLinkTo[1]=-1;
				RoadNetworkRes[i].RoadLinkTo[1]=-1;

				RoadNetworkRes[i].NodeLinkTo[2]=i+RowSize;
				RoadNetworkRes[i].RoadLinkTo[2]=0;

				RoadNetworkRes[i].NodeLinkTo[3]=i+1;
				RoadNetworkRes[i].RoadLinkTo[3]=1;

			}
		}
		else if (i%RowSize==RowSize-1)
			//对应矩阵最后一列元素
		{

			if (i==RowSize-1)
			{
				RoadNetworkRes[i].NodeLinkTo[0]=-1;
				RoadNetworkRes[i].RoadLinkTo[0]=-1;

				RoadNetworkRes[i].NodeLinkTo[1]=i-1;
				RoadNetworkRes[i].RoadLinkTo[1]=3;

				RoadNetworkRes[i].NodeLinkTo[2]=i+RowSize;
				RoadNetworkRes[i].RoadLinkTo[2]=0;

				RoadNetworkRes[i].NodeLinkTo[3]=-1;
				RoadNetworkRes[i].RoadLinkTo[3]=-1;	

			}
			else if (i==RowSize*RowSize-1)
			{
				RoadNetworkRes[i].NodeLinkTo[0]=i-RowSize;
				RoadNetworkRes[i].RoadLinkTo[0]=2;

				RoadNetworkRes[i].NodeLinkTo[1]=i-1;
				RoadNetworkRes[i].RoadLinkTo[1]=3;

				RoadNetworkRes[i].NodeLinkTo[2]=-1;
				RoadNetworkRes[i].RoadLinkTo[2]=-1;

				RoadNetworkRes[i].NodeLinkTo[3]=-1;
				RoadNetworkRes[i].RoadLinkTo[3]=-1;	

			}
			else
			{
				RoadNetworkRes[i].NodeLinkTo[0]=i-RowSize;
				RoadNetworkRes[i].RoadLinkTo[0]=2;

				RoadNetworkRes[i].NodeLinkTo[1]=i-1;
				RoadNetworkRes[i].RoadLinkTo[1]=3;

				RoadNetworkRes[i].NodeLinkTo[2]=i+RowSize;
				RoadNetworkRes[i].RoadLinkTo[2]=0;

				RoadNetworkRes[i].NodeLinkTo[3]=-1;
				RoadNetworkRes[i].RoadLinkTo[3]=-1;	

			}
		}
		else if (i<RowSize && i>0)
			//对应矩阵第一行元素
		{
			RoadNetworkRes[i].NodeLinkTo[0]=-1;
			RoadNetworkRes[i].RoadLinkTo[0]=-1;

			RoadNetworkRes[i].NodeLinkTo[1]=i-1;
			RoadNetworkRes[i].RoadLinkTo[1]=3;

			RoadNetworkRes[i].NodeLinkTo[2]=i+RowSize;
			RoadNetworkRes[i].RoadLinkTo[2]=0;

			RoadNetworkRes[i].NodeLinkTo[3]=i+1;
			RoadNetworkRes[i].RoadLinkTo[3]=1;	

		}
		else if (i<RowSize*RowSize-1 && i>RowSize*(RowSize-1))
			//对应矩阵最后一行元素
		{
			RoadNetworkRes[i].NodeLinkTo[0]=i-RowSize;
			RoadNetworkRes[i].RoadLinkTo[0]=2;

			RoadNetworkRes[i].NodeLinkTo[1]=i-1;
			RoadNetworkRes[i].RoadLinkTo[1]=3;

			RoadNetworkRes[i].NodeLinkTo[2]=-1;
			RoadNetworkRes[i].RoadLinkTo[2]=-1;

			RoadNetworkRes[i].NodeLinkTo[3]=i+1;
			RoadNetworkRes[i].RoadLinkTo[3]=1;	
		}
		else
			//除矩阵边缘外的所有元素
		{
			RoadNetworkRes[i].NodeLinkTo[0]=i-RowSize;
			RoadNetworkRes[i].RoadLinkTo[0]=2;

			RoadNetworkRes[i].NodeLinkTo[1]=i-1;
			RoadNetworkRes[i].RoadLinkTo[1]=3;

			RoadNetworkRes[i].NodeLinkTo[2]=i+RowSize;
			RoadNetworkRes[i].RoadLinkTo[2]=0;

			RoadNetworkRes[i].NodeLinkTo[3]=i+1;
			RoadNetworkRes[i].RoadLinkTo[3]=1;	
		}
	}

	

}

void TrafficLightsDataInitial()
//交通灯数据初始化
{
	for (int l=0;l<NodeNumTol;l++)
	{
		//四相交通灯初始化
		RoadNetworkRes[l].TrafficLight.cycle=Cmax;

		RoadNetworkRes[l].TrafficLight.GreenDuration__Phase1=Cmax/4;
		RoadNetworkRes[l].TrafficLight.GreenDuration__Phase2=Cmax/4;
		RoadNetworkRes[l].TrafficLight.GreenDuration__Phase3=Cmax/4;
		RoadNetworkRes[l].TrafficLight.GreenDuration__Phase4=Cmax/4;

		RoadNetworkRes[l].TrafficLight.Phase=0;
		
		RoadNetworkRes[l].TrafficLight.GreenLightStatus_Phase1=false;
		RoadNetworkRes[l].TrafficLight.GreenLightStatus_Phase2=false;
		RoadNetworkRes[l].TrafficLight.GreenLightStatus_Phase3=false;
		RoadNetworkRes[l].TrafficLight.GreenLightStatus_Phase4=false;
	}

}

void RoadNetworkDataInitial()
//路网数据初始化
{
	for (int i=0;i<NodeNumTol;i++)
	{
		//if (i==12)
		{
			//为各道路车流量赋值
			RoadNetworkRes[i].VehicleArrRate[0]=flow;
			
			RoadNetworkRes[i].VehicleArrRate[1]=flow;
		
			RoadNetworkRes[i].VehicleArrRate[2]=flow;
			
			RoadNetworkRes[i].VehicleArrRate[3]=flow;
			
		}
		//第一场景测试--路网中心点为发车点
		/*if(i==7)
		{
			RoadNetworkRes[i].VehicleArrRate[2]=flow;
		}
		if(i==11)
		{
			RoadNetworkRes[i].VehicleArrRate[3]=flow;
		}
		if(i==17)
		{
			RoadNetworkRes[i].VehicleArrRate[0]=flow;
		}
		if(i==13)
		{
			RoadNetworkRes[i].VehicleArrRate[1]=flow;
		}*/
		//第二场景测试--路网顶点为发车点
		
		
		//各道路长度赋值
		RoadNetworkRes[i].RoadLength[0]=RoadSIZE*VehicleLength;
		RoadNetworkRes[i].RoadLength[1]=RoadSIZE*VehicleLength;
		RoadNetworkRes[i].RoadLength[2]=RoadSIZE*VehicleLength;
		RoadNetworkRes[i].RoadLength[3]=RoadSIZE*VehicleLength;
		
		//各道路通过的车辆数
		RoadNetworkRes[i].RoadVehiclePassNum[0][0]=0;
		RoadNetworkRes[i].RoadVehiclePassNum[0][1]=0;
		RoadNetworkRes[i].RoadVehiclePassNum[0][2]=0;
		RoadNetworkRes[i].RoadVehiclePassNum[1][0]=0;
		RoadNetworkRes[i].RoadVehiclePassNum[1][1]=0;
		RoadNetworkRes[i].RoadVehiclePassNum[1][2]=0;
		RoadNetworkRes[i].RoadVehiclePassNum[2][0]=0;
		RoadNetworkRes[i].RoadVehiclePassNum[2][1]=0;
		RoadNetworkRes[i].RoadVehiclePassNum[2][2]=0;
		RoadNetworkRes[i].RoadVehiclePassNum[3][0]=0;
		RoadNetworkRes[i].RoadVehiclePassNum[3][1]=0;
		RoadNetworkRes[i].RoadVehiclePassNum[3][2]=0;

		//各道路车辆的总行驶时间RoadTraveTime
		RoadNetworkRes[i].RoadTraveTime[0]=0;
		RoadNetworkRes[i].RoadTraveTime[1]=0;
		RoadNetworkRes[i].RoadTraveTime[2]=0;
		RoadNetworkRes[i].RoadTraveTime[3]=0;

		//各道路流入车数赋值
		RoadNetworkRes[i].VehicleInNumInAll[0]=0;
		RoadNetworkRes[i].VehicleInNumInAll[1]=0;
		RoadNetworkRes[i].VehicleInNumInAll[2]=0;
		RoadNetworkRes[i].VehicleInNumInAll[3]=0;

		//各道路流入最大车数赋值
		RoadNetworkRes[i].VehicleInNumInMax[0]=0;
		RoadNetworkRes[i].VehicleInNumInMax[1]=0;
		RoadNetworkRes[i].VehicleInNumInMax[2]=0;
		RoadNetworkRes[i].VehicleInNumInMax[3]=0;

		//各道路的关联度数据
		RoadNetworkRes[i].RoadRelationNum[0]=0;
		RoadNetworkRes[i].RoadRelationNum[1]=0;
		RoadNetworkRes[i].RoadRelationNum[2]=0;
		RoadNetworkRes[i].RoadRelationNum[3]=0;
	}


	//道路车辆数据赋值
	for (int i=0;i<NodeNumTol;i++)
	{
		for (int r=0;r<NumofRoads;r++)
		{
			for (int l=0;l<NumofLanes;l++)
			{
				RoadNetworkRes[i].Road[r][l].RoadFront=0;
				RoadNetworkRes[i].Road[r][l].RoadRear=0;
				RoadNetworkRes[i].Road[r][l].RoadCount=0;

				for (int v=0;v<RoadSIZE;v++)
				{
					RoadNetworkRes[i].Road[r][l].BufferRoad[v].VehicleID=v;
					RoadNetworkRes[i].Road[r][l].BufferRoad[v].RoadTimeConsumed=0;

					RoadNetworkRes[i].Road[r][l].BufferRoad[v].position=-1;
					RoadNetworkRes[i].Road[r][l].BufferRoad[v].speed=0;
					RoadNetworkRes[i].Road[r][l].BufferRoad[v].acceleration=0;
					RoadNetworkRes[i].Road[r][l].BufferRoad[v].VehicleFlag=false;
					RoadNetworkRes[i].Road[r][l].BufferRoad[v].ExpSpeed=60/3.6+((80-30)/3.6)*((float)rand()/RAND_MAX);
				}
			}
		}
		
	}


	for (int i=0;i<NodeNumTol;i++)
	{
		for (int j=0;j<NodeNumTol;j++)
		{
			LapMatirx[i][j]=-1;
		}
	}

	for (int i=0;i<NodeNumTol;i++)
	{
		
		Paint[i]=0;
	
	}


}

void TrafficControlProGen2()
{
	for (int i=0;i<NodeNumTol;i++)
	{
		RoadNetworkRes[i].TrafficLight.cycle=Cmax;
		if (i==0)
		{
			RoadNetworkRes[i].TrafficLight.Phase=0;
		}
		else
		{
			RoadNetworkRes[i].TrafficLight.Phase=Cmax*((float)rand()/RAND_MAX);
		}
		
		RoadNetworkRes[i].TrafficLight.GreenDuration__Phase1=Cmax/4;
		RoadNetworkRes[i].TrafficLight.GreenDuration__Phase2=Cmax/4;
		RoadNetworkRes[i].TrafficLight.GreenDuration__Phase3=Cmax/4;
		RoadNetworkRes[i].TrafficLight.GreenDuration__Phase4=Cmax/4;

	}
}

void TrafficControlProGen()  //生成交通灯控制方案
{
	srand(time(0));
	int C=Cmin+((Cmax-Cmin)*((float)rand()/RAND_MAX));
	for (int i=0;i<NodeNumTol;i++)
	{
		//float a=(float)rand()/RAND_MAX;
		
		RoadNetworkRes[i].TrafficLight.cycle=C;

		if (i==0)
		{
			RoadNetworkRes[i].TrafficLight.Phase=0;
		}
		else
		{
			RoadNetworkRes[i].TrafficLight.Phase=C*((float)rand()/RAND_MAX);
		}

		

		//第一相绿灯持续时间
		RoadNetworkRes[i].TrafficLight.GreenDuration__Phase1=GreenTimeInPhaseMin+((GreenTimeInPhaseMax-GreenTimeInPhaseMin)*((float)rand()/RAND_MAX));
		
		//第二相绿灯持续时间
		int temp = RoadNetworkRes[i].TrafficLight.cycle - RoadNetworkRes[i].TrafficLight.GreenDuration__Phase1;
		if (temp > GreenTimeInPhaseMax)
		{
			RoadNetworkRes[i].TrafficLight.GreenDuration__Phase2 = GreenTimeInPhaseMin + ((GreenTimeInPhaseMax-GreenTimeInPhaseMin) * ((float)rand()/RAND_MAX));
		}
		else
		{
			RoadNetworkRes[i].TrafficLight.GreenDuration__Phase2 = GreenTimeInPhaseMin + ((temp - GreenTimeInPhaseMin)*((float)rand()/RAND_MAX));
		}
		
		//第三相绿灯持续时间
		temp = RoadNetworkRes[i].TrafficLight.cycle-RoadNetworkRes[i].TrafficLight.GreenDuration__Phase1-RoadNetworkRes[i].TrafficLight.GreenDuration__Phase2;
		if (temp>GreenTimeInPhaseMax)
		{
			RoadNetworkRes[i].TrafficLight.GreenDuration__Phase3=GreenTimeInPhaseMin+((GreenTimeInPhaseMax-GreenTimeInPhaseMin)*((float)rand()/RAND_MAX));
		}
		else
		{
			RoadNetworkRes[i].TrafficLight.GreenDuration__Phase3=GreenTimeInPhaseMin+((temp-GreenTimeInPhaseMin)*((float)rand()/RAND_MAX));
		}

		//第四相绿灯持续时间
		int SumThree=RoadNetworkRes[i].TrafficLight.GreenDuration__Phase1+RoadNetworkRes[i].TrafficLight.GreenDuration__Phase2+RoadNetworkRes[i].TrafficLight.GreenDuration__Phase3;
		RoadNetworkRes[i].TrafficLight.GreenDuration__Phase4=RoadNetworkRes[i].TrafficLight.cycle - SumThree;
	}
}

void TrafficLightsRefresh()
//交通灯信号刷新
{
	for (int i = 0; i < NodeNumTol; i++)
	{
		if(i==0)
		{
			int C=RoadNetworkRes[i].TrafficLight.cycle;
			int t1,t2,t3,t4;
			t1=RoadNetworkRes[i].TrafficLight.GreenDuration__Phase1;
			t2=RoadNetworkRes[i].TrafficLight.GreenDuration__Phase2;
			t3=RoadNetworkRes[i].TrafficLight.GreenDuration__Phase3;
			t4=RoadNetworkRes[i].TrafficLight.GreenDuration__Phase4;

			if (SimulationClock%C>=0 && SimulationClock%C<=t1)
			{
				RoadNetworkRes[i].TrafficLight.GreenLightStatus_Phase1=true;
				RoadNetworkRes[i].TrafficLight.GreenLightStatus_Phase2=false;
				RoadNetworkRes[i].TrafficLight.GreenLightStatus_Phase3=false;
				RoadNetworkRes[i].TrafficLight.GreenLightStatus_Phase4=false;
			}
			if (SimulationClock%C>t1 && SimulationClock%C<=t1+t2)
			{
				RoadNetworkRes[i].TrafficLight.GreenLightStatus_Phase1=false;
				RoadNetworkRes[i].TrafficLight.GreenLightStatus_Phase2=true;
				RoadNetworkRes[i].TrafficLight.GreenLightStatus_Phase3=false;
				RoadNetworkRes[i].TrafficLight.GreenLightStatus_Phase4=false;
			}
			if (SimulationClock%C>t1+t2 && SimulationClock%C<=t1+t2+t3)
			{
				RoadNetworkRes[i].TrafficLight.GreenLightStatus_Phase1=false;
				RoadNetworkRes[i].TrafficLight.GreenLightStatus_Phase2=false;
				RoadNetworkRes[i].TrafficLight.GreenLightStatus_Phase3=true;
				RoadNetworkRes[i].TrafficLight.GreenLightStatus_Phase4=false;
			}
			if (SimulationClock%C>t1+t2+t3 && SimulationClock%C<=t1+t2+t3+t4)
			{
				RoadNetworkRes[i].TrafficLight.GreenLightStatus_Phase1=false;
				RoadNetworkRes[i].TrafficLight.GreenLightStatus_Phase2=false;
				RoadNetworkRes[i].TrafficLight.GreenLightStatus_Phase3=false;
				RoadNetworkRes[i].TrafficLight.GreenLightStatus_Phase4=true;
			}
		}
		else
		{
			int C=RoadNetworkRes[i].TrafficLight.cycle;
			int phase=RoadNetworkRes[i].TrafficLight.Phase;
			int t1,t2,t3,t4;
			t1=RoadNetworkRes[i].TrafficLight.GreenDuration__Phase1;
			t2=RoadNetworkRes[i].TrafficLight.GreenDuration__Phase2;
			t3=RoadNetworkRes[i].TrafficLight.GreenDuration__Phase3;
			t4=RoadNetworkRes[i].TrafficLight.GreenDuration__Phase4;

			if (SimulationClock<phase)
			{
				RoadNetworkRes[i].TrafficLight.GreenLightStatus_Phase1=false;
				RoadNetworkRes[i].TrafficLight.GreenLightStatus_Phase2=false;
				RoadNetworkRes[i].TrafficLight.GreenLightStatus_Phase3=false;
				RoadNetworkRes[i].TrafficLight.GreenLightStatus_Phase4=true;
			}
			else
			{
				if ((SimulationClock-phase)%C>=0 && (SimulationClock-phase)%C<=t1)
				{
					RoadNetworkRes[i].TrafficLight.GreenLightStatus_Phase1=true;
					RoadNetworkRes[i].TrafficLight.GreenLightStatus_Phase2=false;
					RoadNetworkRes[i].TrafficLight.GreenLightStatus_Phase3=false;
					RoadNetworkRes[i].TrafficLight.GreenLightStatus_Phase4=false;
				}
				if ((SimulationClock-phase)%C>t1 && (SimulationClock-phase)%C<=t1+t2)
				{
					RoadNetworkRes[i].TrafficLight.GreenLightStatus_Phase1=false;
					RoadNetworkRes[i].TrafficLight.GreenLightStatus_Phase2=true;
					RoadNetworkRes[i].TrafficLight.GreenLightStatus_Phase3=false;
					RoadNetworkRes[i].TrafficLight.GreenLightStatus_Phase4=false;
				}
				if ((SimulationClock-phase)%C>t1+t2 && (SimulationClock-phase)%C<=t1+t2+t3)
				{
					RoadNetworkRes[i].TrafficLight.GreenLightStatus_Phase1=false;
					RoadNetworkRes[i].TrafficLight.GreenLightStatus_Phase2=false;
					RoadNetworkRes[i].TrafficLight.GreenLightStatus_Phase3=true;
					RoadNetworkRes[i].TrafficLight.GreenLightStatus_Phase4=false;
				}
				if ((SimulationClock-phase)%C>t1+t2+t3 && (SimulationClock-phase)%C<=t1+t2+t3+t4)
				{
					RoadNetworkRes[i].TrafficLight.GreenLightStatus_Phase1=false;
					RoadNetworkRes[i].TrafficLight.GreenLightStatus_Phase2=false;
					RoadNetworkRes[i].TrafficLight.GreenLightStatus_Phase3=false;
					RoadNetworkRes[i].TrafficLight.GreenLightStatus_Phase4=true;
				}
			}
		}
	}
}

int VehGenEvent[NodeNumTol][NumofRoads][NumofLanes];//发生车辆生成事件的时间点
double interval[NodeNumTol][NumofRoads][NumofLanes];
int intervalReal[NodeNumTol][NumofRoads][NumofLanes];

float RandomGen(float Lambda)
{
	float Z;
	float output;

	do 
	{
		Z=((float)rand()/RAND_MAX);//Z为0-1之间均匀分布随机数
	} 
	while ((Z==0)||(Z==1));

	float Z1=log(Z);
	float X=1/Lambda;
	output=-X*Z1;

	return (output);
}

double GaussRandomGen(float mu, float sigma)
{
	float u1,u2;
	float T1,T2;
	float output1;
	float R=1000;
	while(R>=1.0||R==0)
	{
		do 
		{
			u1=((float)rand()/RAND_MAX);//u1为0-1之间均匀分布随机数
			u2=((float)rand()/RAND_MAX);//u2为0-1之间均匀分布随机数
		} 
		while (u1==u2);
		T1=2*u1-1;
		T2=2*u2-1;

		R=T1*T1+T2*T2;
	}
	float factor=sqrt((-2)*log(R)/R);
	output1=mu+sigma*T1*factor;

	return output1;
}


void VehicleGeneration()
//车辆生成
{
	srand(int(time(0)));//初次生成随机数
	float randn;//定义大小
	randn=rand();

	for (int n=0;n<NodeNumTol;n++)
	{
		for (int r=0;r<NumofRoads;r++)
		{
			for (int l=0;l<NumofLanes;l++)
			{
				if (SimulationClock==0)//设定仿真开始时车数目为零
				{
					srand(unsigned(randn));
					randn=rand();

					float VehArriveRate=RoadNetworkRes[n].VehicleArrRate[r];//三条车道共享同一车辆到达数据

					interval[n][r][l]=RandomGen(VehArriveRate);	
					intervalReal[n][r][l]=int(interval[n][r][l]+1);//转换为整形数据
					VehGenEvent[n][r][l]=SimulationClock+intervalReal[n][r][l];//计算下一辆车到达的时间

				}
				else
				{
					if (VehGenEvent[n][r][l]==SimulationClock)//遇到车辆进入事件
					{
						//****(1) 在这里需要实例化车辆对象，并加入道路队列
						srand(unsigned(randn));
						randn=rand();
						//生成车辆
						VehicleAgent AgentCar;
						AgentCar.position=0;//路段总长度为1000米，初始发车位置为0！注意要区别于初始化位置-1
						AgentCar.speed=10;//float(abs(GaussRandomGen(40/3.6,0.0001)));//速度平均值为40km/小时
						AgentCar.acceleration=0;//设定初始加速度为0，即匀速进入，这里也可以符合高斯分布进入				
						AgentCar.ExpSpeed=60/3.6+(20/3.6)*((float)rand()/RAND_MAX);

						AgentCar.RoadTimeConsumed=0;
						int temp2;
						temp2=RoadNetworkRes[n].NodeLinkTo[r];

						bool yesornot;
						// wangkai
						if ((n==1&&r==1)||(n==5&&r==0)||(n==3&&r==3)||(n==9&&r==0)||(n==15&&r==2)||(n==21&&r==1)||(n==19&&r==2)||(n==23&&r==3))//(n==7&&r==2)||(n==11&&r==3)||(n==13&&r==1)||(n==17&&r==0)/
						{

							int rear=RoadNetworkRes[n].Road[r][l].RoadRear;
							int front=RoadNetworkRes[n].Road[r][l].RoadFront;
							if ((rear+1)%RoadSIZE!=front)
							{
								RoadNetworkRes[n].Road[r][l].BufferRoad[rear].position=AgentCar.position;
								RoadNetworkRes[n].Road[r][l].BufferRoad[rear].speed=AgentCar.speed;
								RoadNetworkRes[n].Road[r][l].BufferRoad[rear].acceleration=AgentCar.acceleration;
								RoadNetworkRes[n].Road[r][l].BufferRoad[rear].ExpSpeed=AgentCar.ExpSpeed;
								RoadNetworkRes[n].Road[r][l].BufferRoad[rear].RoadTimeConsumed=AgentCar.RoadTimeConsumed;
								RoadNetworkRes[n].Road[r][l].BufferRoad[rear].VehicleFlag=true;
								RoadNetworkRes[n].Road[r][l].RoadRear=(RoadNetworkRes[n].Road[r][l].RoadRear+1)%RoadSIZE;//循环队列的操作
								RoadNetworkRes[n].Road[r][l].RoadCount++;
							}

						}
						//****(2) 生成下一次车辆到达的时间
						float VehArriveRate=RoadNetworkRes[n].VehicleArrRate[r];//三条车道共享同一车辆到达数据
						interval[n][r][l]=RandomGen(VehArriveRate);
						intervalReal[n][r][l]=int(interval[n][r][l]+1);//转换为整形数据
						VehGenEvent[n][r][l]=SimulationClock+intervalReal[n][r][l];
					}
					else
					{
						VehGenEvent[n][r][l]=VehGenEvent[n][r][l];
					}
				}

			}
		}
	}
}



void SimulationProcess()
//仿真主体函数
{
	float positionNew,speedNew,accelerationNew;
	bool flagNew;

	for (int n=0;n<NodeNumTol;n++)
	{
		for (int r=0;r<NumofRoads;r++)
		{
			for (int l=0;l<NumofLanes;l++)
			{
				for (int v=0;v<RoadSIZE;v++)
				{
				
					positionNew=RoadNetworkRes[n].Road[r][l].BufferRoad[v].position;
					speedNew=RoadNetworkRes[n].Road[r][l].BufferRoad[v].speed;
					accelerationNew=RoadNetworkRes[n].Road[r][l].BufferRoad[v].acceleration;
					flagNew=RoadNetworkRes[n].Road[r][l].BufferRoad[v].VehicleFlag;

					if (RoadNetworkRes[n].Road[r][l].BufferRoad[v].VehicleFlag==true)
					{
						if (RoadNetworkRes[n].Road[r][l].BufferRoad[v].ExpSpeed==0)
						{
							int a=0;
						}


						//！非头车状态更新1
						if(v>0 && RoadNetworkRes[n].Road[r][l].BufferRoad[v-1].VehicleFlag==true)
						{
							if (RoadNetworkRes[n].Road[r][l].BufferRoad[v-1].position<RoadNetworkRes[n].Road[r][l].BufferRoad[v].position) //不合理保护
							{
								positionNew=RoadNetworkRes[n].Road[r][l].BufferRoad[v].position;
								speedNew=0;
								accelerationNew=0;
							}
							if (RoadNetworkRes[n].Road[r][l].BufferRoad[v-1].position-RoadNetworkRes[n].Road[r][l].BufferRoad[v].position<=150 && RoadNetworkRes[n].Road[r][l].BufferRoad[v-1].position-RoadNetworkRes[n].Road[r][l].BufferRoad[v].position>0)
								//跟驰状态
							{
								if (RoadNetworkRes[n].Road[r][l].BufferRoad[v].position==0)//刚发车的车辆
								{
									if (RoadNetworkRes[n].Road[r][l].BufferRoad[v-1].position>0) //前车已处于运行状态，本次发车需要避免初始速度过大
									{
										while(RoadNetworkRes[n].Road[r][l].BufferRoad[v].speed*1>RoadNetworkRes[n].Road[r][l].BufferRoad[v-1].position)
											//调整发车速度
										{
											RoadNetworkRes[n].Road[r][l].BufferRoad[v].speed=RoadNetworkRes[n].Road[r][l].BufferRoad[v].speed/2;
										}
										//更新加速度
										if(RoadNetworkRes[n].Road[r][l].BufferRoad[v].speed==0 && RoadNetworkRes[n].Road[r][l].BufferRoad[v-1].position-RoadNetworkRes[n].Road[r][l].BufferRoad[v].position>5)
											accelerationNew=RoadNetworkRes[n].Road[r][l].BufferRoad[v].acceleration+1;
										else
											accelerationNew=RoadNetworkRes[n].Road[r][l].BufferRoad[v].speed*(RoadNetworkRes[n].Road[r][l].BufferRoad[v-1].speed-RoadNetworkRes[n].Road[r][l].BufferRoad[v].speed)/(RoadNetworkRes[n].Road[r][l].BufferRoad[v-1].position-RoadNetworkRes[n].Road[r][l].BufferRoad[v].position);

										//更新速度
										speedNew=RoadNetworkRes[n].Road[r][l].BufferRoad[v].speed+RoadNetworkRes[n].Road[r][l].BufferRoad[v].acceleration;
										if (speedNew<0)
										{
											speedNew=0;
											accelerationNew=0;
										}

										//更新位置
										positionNew=RoadNetworkRes[n].Road[r][l].BufferRoad[v].position+RoadNetworkRes[n].Road[r][l].BufferRoad[v].speed+0.5*RoadNetworkRes[n].Road[r][l].BufferRoad[v].acceleration;
										if (positionNew<RoadNetworkRes[n].Road[r][l].BufferRoad[v].position || positionNew>=RoadNetworkRes[n].Road[r][l].BufferRoad[v-1].position+RoadNetworkRes[n].Road[r][l].BufferRoad[v-1].speed+0.5*RoadNetworkRes[n].Road[r][l].BufferRoad[v-1].acceleration*RoadNetworkRes[n].Road[r][l].BufferRoad[v-1].acceleration)
										{
											positionNew=RoadNetworkRes[n].Road[r][l].BufferRoad[v].position;
											speedNew=0;
											accelerationNew=0;
										}
									}
									else//前车还未发车，本车保持初始状态不动
									{
										positionNew=RoadNetworkRes[n].Road[r][l].BufferRoad[v].position;
										speedNew=0;
										accelerationNew=0;
									}
								}
								else//非刚发车车辆
								{
									//更新加速度
									if(RoadNetworkRes[n].Road[r][l].BufferRoad[v].speed==0 && RoadNetworkRes[n].Road[r][l].BufferRoad[v-1].position-RoadNetworkRes[n].Road[r][l].BufferRoad[v].position>5)
										accelerationNew=RoadNetworkRes[n].Road[r][l].BufferRoad[v].acceleration+1;
									else
										accelerationNew=RoadNetworkRes[n].Road[r][l].BufferRoad[v].speed*(RoadNetworkRes[n].Road[r][l].BufferRoad[v-1].speed-RoadNetworkRes[n].Road[r][l].BufferRoad[v].speed)/(RoadNetworkRes[n].Road[r][l].BufferRoad[v-1].position-RoadNetworkRes[n].Road[r][l].BufferRoad[v].position);

									//更新速度
									speedNew=RoadNetworkRes[n].Road[r][l].BufferRoad[v].speed+RoadNetworkRes[n].Road[r][l].BufferRoad[v].acceleration;
									if (speedNew<0)
									{
										speedNew=0;
										accelerationNew=0;
									}

									//更新位置
									positionNew=RoadNetworkRes[n].Road[r][l].BufferRoad[v].position+RoadNetworkRes[n].Road[r][l].BufferRoad[v].speed+0.5*RoadNetworkRes[n].Road[r][l].BufferRoad[v].acceleration;
									if (positionNew<RoadNetworkRes[n].Road[r][l].BufferRoad[v].position || positionNew>=RoadNetworkRes[n].Road[r][l].BufferRoad[v-1].position+RoadNetworkRes[n].Road[r][l].BufferRoad[v-1].speed+0.5*RoadNetworkRes[n].Road[r][l].BufferRoad[v-1].acceleration*RoadNetworkRes[n].Road[r][l].BufferRoad[v-1].acceleration)
									{
										positionNew=RoadNetworkRes[n].Road[r][l].BufferRoad[v].position;
										speedNew=0;
										accelerationNew=0;
									}
								}
							}
							if(RoadNetworkRes[n].Road[r][l].BufferRoad[v-1].position-RoadNetworkRes[n].Road[r][l].BufferRoad[v].position>150)
								//自由行驶
							{
								//更新加速度
								if(RoadNetworkRes[n].Road[r][l].BufferRoad[v].speed<RoadNetworkRes[n].Road[r][l].BufferRoad[v].ExpSpeed)
									accelerationNew=RoadNetworkRes[n].Road[r][l].BufferRoad[v].acceleration+1;
								else
								{
									speedNew=RoadNetworkRes[n].Road[r][l].BufferRoad[v].ExpSpeed;
									accelerationNew=0;
								}


								//更新速度
								speedNew=RoadNetworkRes[n].Road[r][l].BufferRoad[v].speed+RoadNetworkRes[n].Road[r][l].BufferRoad[v].acceleration;
								if (speedNew<0)
								{
									speedNew=0;
									accelerationNew=0;
								}

								//更新位置
								positionNew=RoadNetworkRes[n].Road[r][l].BufferRoad[v].position+RoadNetworkRes[n].Road[r][l].BufferRoad[v].speed+0.5*RoadNetworkRes[n].Road[r][l].BufferRoad[v].acceleration;
								if (positionNew<RoadNetworkRes[n].Road[r][l].BufferRoad[v].position || positionNew>=RoadNetworkRes[n].Road[r][l].BufferRoad[v-1].position+RoadNetworkRes[n].Road[r][l].BufferRoad[v-1].speed+0.5*RoadNetworkRes[n].Road[r][l].BufferRoad[v-1].acceleration*RoadNetworkRes[n].Road[r][l].BufferRoad[v-1].acceleration)
								{
									positionNew=RoadNetworkRes[n].Road[r][l].BufferRoad[v].position;
									speedNew=0;
									accelerationNew=0;
								}
							}

						}





						//！！非头车状态更新法则2
						if (v==0 && RoadNetworkRes[n].Road[r][l].BufferRoad[RoadSIZE-1].VehicleFlag==true)
						{
							if (RoadNetworkRes[n].Road[r][l].BufferRoad[RoadSIZE-1].position<RoadNetworkRes[n].Road[r][l].BufferRoad[v].position)
								//不合理保护
							{
								positionNew=RoadNetworkRes[n].Road[r][l].BufferRoad[v].position;
								speedNew=0;
								accelerationNew=0;
							}
							if (RoadNetworkRes[n].Road[r][l].BufferRoad[RoadSIZE-1].position-RoadNetworkRes[n].Road[r][l].BufferRoad[v].position<=150 && RoadNetworkRes[n].Road[r][l].BufferRoad[RoadSIZE-1].position-RoadNetworkRes[n].Road[r][l].BufferRoad[v].position>0)
								//跟驰状态
							{
								if (RoadNetworkRes[n].Road[r][l].BufferRoad[v].position==0)//刚发车的车辆
								{
									if (RoadNetworkRes[n].Road[r][l].BufferRoad[RoadSIZE-1].position>0)//前车已处于运行状态，本次发车需要避免初始速度过大
									{
										while(RoadNetworkRes[n].Road[r][l].BufferRoad[v].speed*1>RoadNetworkRes[n].Road[r][l].BufferRoad[RoadSIZE-1].position)
											//调整发车速度
										{
											RoadNetworkRes[n].Road[r][l].BufferRoad[v].speed=RoadNetworkRes[n].Road[r][l].BufferRoad[v].speed/2;
										}
										//更新加速度
										if(RoadNetworkRes[n].Road[r][l].BufferRoad[v].speed==0 && RoadNetworkRes[n].Road[r][l].BufferRoad[RoadSIZE-1].position-RoadNetworkRes[n].Road[r][l].BufferRoad[v].position>5)
											accelerationNew=RoadNetworkRes[n].Road[r][l].BufferRoad[v].acceleration+1;
										else
											accelerationNew=RoadNetworkRes[n].Road[r][l].BufferRoad[v].speed*(RoadNetworkRes[n].Road[r][l].BufferRoad[RoadSIZE-1].speed-RoadNetworkRes[n].Road[r][l].BufferRoad[v].speed)/(RoadNetworkRes[n].Road[r][l].BufferRoad[RoadSIZE-1].position-RoadNetworkRes[n].Road[r][l].BufferRoad[v].position);

										//更新速度
										speedNew=RoadNetworkRes[n].Road[r][l].BufferRoad[v].speed+RoadNetworkRes[n].Road[r][l].BufferRoad[v].acceleration;
										if (speedNew<0)
										{
											speedNew=0;
											accelerationNew=0;
										}

										//更新位置
										positionNew=RoadNetworkRes[n].Road[r][l].BufferRoad[v].position+RoadNetworkRes[n].Road[r][l].BufferRoad[v].speed+0.5*RoadNetworkRes[n].Road[r][l].BufferRoad[v].acceleration;
										if (positionNew<RoadNetworkRes[n].Road[r][l].BufferRoad[v].position || positionNew>=RoadNetworkRes[n].Road[r][l].BufferRoad[RoadSIZE-1].position+RoadNetworkRes[n].Road[r][l].BufferRoad[RoadSIZE-1].speed+0.5*RoadNetworkRes[n].Road[r][l].BufferRoad[RoadSIZE-1].acceleration*RoadNetworkRes[n].Road[r][l].BufferRoad[RoadSIZE-1].acceleration)
										{
											positionNew=RoadNetworkRes[n].Road[r][l].BufferRoad[v].position;
											speedNew=0;
											accelerationNew=0;
										}
									}
									else//前车还未发车，本车保持初始状态不动
									{
										positionNew=RoadNetworkRes[n].Road[r][l].BufferRoad[v].position;
										speedNew=0;
										accelerationNew=0;
									}
								}
								else//非刚发车车辆
								{
									//更新加速度
									if(RoadNetworkRes[n].Road[r][l].BufferRoad[v].speed==0 && RoadNetworkRes[n].Road[r][l].BufferRoad[RoadSIZE-1].position-RoadNetworkRes[n].Road[r][l].BufferRoad[v].position>5)
										accelerationNew=RoadNetworkRes[n].Road[r][l].BufferRoad[v].acceleration+1;
									else
										accelerationNew=RoadNetworkRes[n].Road[r][l].BufferRoad[v].speed*(RoadNetworkRes[n].Road[r][l].BufferRoad[RoadSIZE-1].speed-RoadNetworkRes[n].Road[r][l].BufferRoad[v].speed)/(RoadNetworkRes[n].Road[r][l].BufferRoad[RoadSIZE-1].position-RoadNetworkRes[n].Road[r][l].BufferRoad[v].position);

									//更新速度
									speedNew=RoadNetworkRes[n].Road[r][l].BufferRoad[v].speed+RoadNetworkRes[n].Road[r][l].BufferRoad[v].acceleration;
									if (speedNew<0)
									{
										speedNew=0;
										accelerationNew=0;
									}

									//更新位置
									positionNew=RoadNetworkRes[n].Road[r][l].BufferRoad[v].position+RoadNetworkRes[n].Road[r][l].BufferRoad[v].speed+0.5*RoadNetworkRes[n].Road[r][l].BufferRoad[v].acceleration;
									if (positionNew<RoadNetworkRes[n].Road[r][l].BufferRoad[v].position || positionNew>=RoadNetworkRes[n].Road[r][l].BufferRoad[RoadSIZE-1].position+RoadNetworkRes[n].Road[r][l].BufferRoad[RoadSIZE-1].speed+0.5*RoadNetworkRes[n].Road[r][l].BufferRoad[RoadSIZE-1].acceleration*RoadNetworkRes[n].Road[r][l].BufferRoad[RoadSIZE-1].acceleration)
									{
										positionNew=RoadNetworkRes[n].Road[r][l].BufferRoad[v].position;
										speedNew=0;
										accelerationNew=0;
									}
								}
							}
							if(RoadNetworkRes[n].Road[r][l].BufferRoad[RoadSIZE-1].position-RoadNetworkRes[n].Road[r][l].BufferRoad[v].position>150)
								//自由行驶
							{
								//更新加速度
								if(RoadNetworkRes[n].Road[r][l].BufferRoad[v].speed<RoadNetworkRes[n].Road[r][l].BufferRoad[v].ExpSpeed)
									accelerationNew=RoadNetworkRes[n].Road[r][l].BufferRoad[v].acceleration+1;
								else
								{
									speedNew=speedNew;
									accelerationNew=0;
								}


								//更新速度
								speedNew=RoadNetworkRes[n].Road[r][l].BufferRoad[v].speed+RoadNetworkRes[n].Road[r][l].BufferRoad[v].acceleration;
								if (speedNew<0)
								{
									speedNew=0;
									accelerationNew=0;
								}

								//更新位置
								positionNew=RoadNetworkRes[n].Road[r][l].BufferRoad[v].position+RoadNetworkRes[n].Road[r][l].BufferRoad[v].speed+0.5*RoadNetworkRes[n].Road[r][l].BufferRoad[v].acceleration;
								if (positionNew<RoadNetworkRes[n].Road[r][l].BufferRoad[v].position || positionNew>=RoadNetworkRes[n].Road[r][l].BufferRoad[RoadSIZE-1].position+RoadNetworkRes[n].Road[r][l].BufferRoad[RoadSIZE-1].speed+0.5*RoadNetworkRes[n].Road[r][l].BufferRoad[RoadSIZE-1].acceleration*RoadNetworkRes[n].Road[r][l].BufferRoad[RoadSIZE-1].acceleration)
								{
									positionNew=RoadNetworkRes[n].Road[r][l].BufferRoad[v].position;
									speedNew=0;
									accelerationNew=0;
								}
							}
						}




						//！！头车状态更新法则
						if ((v>0 && RoadNetworkRes[n].Road[r][l].BufferRoad[v-1].VehicleFlag==false)  || (v==0 && RoadNetworkRes[n].Road[r][l].BufferRoad[RoadSIZE-1].VehicleFlag==false))
						{
							if (RoadNetworkRes[n].TrafficLight.GreenLightStatus_Phase1==true)
							{
								if ((r%2==0 && l==1) || l==2)
								{
									if(RoadNetworkRes[n].Road[r][l].BufferRoad[v].speed<RoadNetworkRes[n].Road[r][l].BufferRoad[v].ExpSpeed)
										accelerationNew=RoadNetworkRes[n].Road[r][l].BufferRoad[v].acceleration+1;
									else

									{
										speedNew=RoadNetworkRes[n].Road[r][l].BufferRoad[v].ExpSpeed;
										accelerationNew=0;//速度达到期望速度后转入匀速行驶
									}


									speedNew=RoadNetworkRes[n].Road[r][l].BufferRoad[v].speed+RoadNetworkRes[n].Road[r][l].BufferRoad[v].acceleration*1;
									if(speedNew<0)
									{
										speedNew=0;
									}
									positionNew=RoadNetworkRes[n].Road[r][l].BufferRoad[v].position+RoadNetworkRes[n].Road[r][l].BufferRoad[v].speed*1+0.5*RoadNetworkRes[n].Road[r][l].BufferRoad[v].acceleration;//时间间隔为1秒	
									if(positionNew<RoadNetworkRes[n].Road[r][l].BufferRoad[v-1].position)
									{
										positionNew=RoadNetworkRes[n].Road[r][l].BufferRoad[v].position;
									}
								}
								else//其他相均为红灯
								{
									if(VehicleLength*RoadSIZE-RoadNetworkRes[n].Road[r][l].BufferRoad[v].position>150)//代表道路总长，起点为0，则道路总长为交通灯所在位置
									{
										if(RoadNetworkRes[n].Road[r][l].BufferRoad[v].speed<RoadNetworkRes[n].Road[r][l].BufferRoad[v].ExpSpeed)
											accelerationNew=RoadNetworkRes[n].Road[r][l].BufferRoad[v].acceleration+1;
										else

										{
											speedNew=RoadNetworkRes[n].Road[r][l].BufferRoad[v].ExpSpeed;
											accelerationNew=0;//速度达到期望速度后转入匀速行驶
										}


										speedNew=RoadNetworkRes[n].Road[r][l].BufferRoad[v].speed+RoadNetworkRes[n].Road[r][l].BufferRoad[v].acceleration*1;
										if(speedNew<0)
										{
											speedNew=0;
										}
										positionNew=RoadNetworkRes[n].Road[r][l].BufferRoad[v].position+RoadNetworkRes[n].Road[r][l].BufferRoad[v].speed*1+0.5*RoadNetworkRes[n].Road[r][l].BufferRoad[v].acceleration;//时间间隔为1秒	
										if(positionNew<RoadNetworkRes[n].Road[r][l].BufferRoad[v-1].position)
										{
											positionNew=RoadNetworkRes[n].Road[r][l].BufferRoad[v].position;
										}
									}
									else
									{
										//根据GM模型更新加速度
										accelerationNew=1*RoadNetworkRes[n].Road[r][l].BufferRoad[v].speed*(0-RoadNetworkRes[n].Road[r][l].BufferRoad[v].speed)/(VehicleLength*RoadSIZE-RoadNetworkRes[n].Road[r][l].BufferRoad[v].position);//1为反映强度系数					

										//更新速度				
										speedNew=RoadNetworkRes[n].Road[r][l].BufferRoad[v].speed+RoadNetworkRes[n].Road[r][l].BufferRoad[v].acceleration*1;
										if(speedNew<0)
										{
											speedNew=0;
										}

										//更新位置
										positionNew=RoadNetworkRes[n].Road[r][l].BufferRoad[v].position+RoadNetworkRes[n].Road[r][l].BufferRoad[v].speed+0.5*RoadNetworkRes[n].Road[r][l].BufferRoad[v].acceleration;//时间间隔为1秒
										if(positionNew<RoadNetworkRes[n].Road[r][l].BufferRoad[v].position)
										{
											positionNew=RoadNetworkRes[n].Road[r][l].BufferRoad[v].position;
										}	
									}
								}
							}
							if (RoadNetworkRes[n].TrafficLight.GreenLightStatus_Phase2==true)
							{
								if ((r%2==0 && l==0) || l==2)
								{
									if(RoadNetworkRes[n].Road[r][l].BufferRoad[v].speed<RoadNetworkRes[n].Road[r][l].BufferRoad[v].ExpSpeed)
										accelerationNew=RoadNetworkRes[n].Road[r][l].BufferRoad[v].acceleration+1;
									else
									{
										speedNew=RoadNetworkRes[n].Road[r][l].BufferRoad[v].ExpSpeed;
										accelerationNew=0;//速度达到期望速度后转入匀速行驶
									}


									speedNew=RoadNetworkRes[n].Road[r][l].BufferRoad[v].speed+RoadNetworkRes[n].Road[r][l].BufferRoad[v].acceleration*1;
									if(speedNew<0)
									{
										speedNew=0;
									}
									positionNew=RoadNetworkRes[n].Road[r][l].BufferRoad[v].position+RoadNetworkRes[n].Road[r][l].BufferRoad[v].speed*1+0.5*RoadNetworkRes[n].Road[r][l].BufferRoad[v].acceleration;//时间间隔为1秒	
									if(positionNew<RoadNetworkRes[n].Road[r][l].BufferRoad[v-1].position)
									{
										positionNew=RoadNetworkRes[n].Road[r][l].BufferRoad[v].position;
									}
								}
								else
								{
									if(VehicleLength*RoadSIZE-RoadNetworkRes[n].Road[r][l].BufferRoad[v].position>150)//代表道路总长，起点为0，则道路总长为交通灯所在位置
									{
										if(RoadNetworkRes[n].Road[r][l].BufferRoad[v].speed<RoadNetworkRes[n].Road[r][l].BufferRoad[v].ExpSpeed)
											accelerationNew=RoadNetworkRes[n].Road[r][l].BufferRoad[v].acceleration+1;
										else

										{
											speedNew=RoadNetworkRes[n].Road[r][l].BufferRoad[v].ExpSpeed;
											accelerationNew=0;//速度达到期望速度后转入匀速行驶
										}


										speedNew=RoadNetworkRes[n].Road[r][l].BufferRoad[v].speed+RoadNetworkRes[n].Road[r][l].BufferRoad[v].acceleration*1;
										if(speedNew<0)
										{
											speedNew=0;
										}
										positionNew=RoadNetworkRes[n].Road[r][l].BufferRoad[v].position+RoadNetworkRes[n].Road[r][l].BufferRoad[v].speed*1+0.5*RoadNetworkRes[n].Road[r][l].BufferRoad[v].acceleration;//时间间隔为1秒	
										if(positionNew<RoadNetworkRes[n].Road[r][l].BufferRoad[v-1].position)
										{
											positionNew=RoadNetworkRes[n].Road[r][l].BufferRoad[v].position;
										}
									}
									else
									{
										//根据GM模型更新加速度
										accelerationNew=1*RoadNetworkRes[n].Road[r][l].BufferRoad[v].speed*(0-RoadNetworkRes[n].Road[r][l].BufferRoad[v].speed)/(VehicleLength*RoadSIZE-RoadNetworkRes[n].Road[r][l].BufferRoad[v].position);//1为反映强度系数					

										//更新速度				
										speedNew=RoadNetworkRes[n].Road[r][l].BufferRoad[v].speed+RoadNetworkRes[n].Road[r][l].BufferRoad[v].acceleration*1;
										if(speedNew<0)
										{
											speedNew=0;
										}

										//更新位置
										positionNew=RoadNetworkRes[n].Road[r][l].BufferRoad[v].position+RoadNetworkRes[n].Road[r][l].BufferRoad[v].speed+0.5*RoadNetworkRes[n].Road[r][l].BufferRoad[v].acceleration;//时间间隔为1秒
										if(positionNew<RoadNetworkRes[n].Road[r][l].BufferRoad[v].position)
										{
											positionNew=RoadNetworkRes[n].Road[r][l].BufferRoad[v].position;
										}	

									}

								}
							}
							if (RoadNetworkRes[n].TrafficLight.GreenLightStatus_Phase3==true)
							{
								if ((r%2!=0 && l==1) || l==2)
								{
									if(RoadNetworkRes[n].Road[r][l].BufferRoad[v].speed<RoadNetworkRes[n].Road[r][l].BufferRoad[v].ExpSpeed)
										accelerationNew=RoadNetworkRes[n].Road[r][l].BufferRoad[v].acceleration+1;
									else
									{
										speedNew=RoadNetworkRes[n].Road[r][l].BufferRoad[v].ExpSpeed;
										accelerationNew=0;//速度达到期望速度后转入匀速行驶
									}


									speedNew=RoadNetworkRes[n].Road[r][l].BufferRoad[v].speed+RoadNetworkRes[n].Road[r][l].BufferRoad[v].acceleration*1;
									if(speedNew<0)
									{
										speedNew=0;
									}
									positionNew=RoadNetworkRes[n].Road[r][l].BufferRoad[v].position+RoadNetworkRes[n].Road[r][l].BufferRoad[v].speed*1+0.5*RoadNetworkRes[n].Road[r][l].BufferRoad[v].acceleration;//时间间隔为1秒	
									if(positionNew<RoadNetworkRes[n].Road[r][l].BufferRoad[v-1].position)
									{
										positionNew=RoadNetworkRes[n].Road[r][l].BufferRoad[v].position;
									}
								}
								else
								{
									if(VehicleLength*RoadSIZE-RoadNetworkRes[n].Road[r][l].BufferRoad[v].position>150)//代表道路总长，起点为0，则道路总长为交通灯所在位置
									{
										if(RoadNetworkRes[n].Road[r][l].BufferRoad[v].speed<RoadNetworkRes[n].Road[r][l].BufferRoad[v].ExpSpeed)
											accelerationNew=RoadNetworkRes[n].Road[r][l].BufferRoad[v].acceleration+1;
										else

										{
											speedNew=RoadNetworkRes[n].Road[r][l].BufferRoad[v].ExpSpeed;
											accelerationNew=0;//速度达到期望速度后转入匀速行驶

										}

										speedNew=RoadNetworkRes[n].Road[r][l].BufferRoad[v].speed+RoadNetworkRes[n].Road[r][l].BufferRoad[v].acceleration*1;
										if(speedNew<0)
										{
											speedNew=0;
										}
										positionNew=RoadNetworkRes[n].Road[r][l].BufferRoad[v].position+RoadNetworkRes[n].Road[r][l].BufferRoad[v].speed*1+0.5*RoadNetworkRes[n].Road[r][l].BufferRoad[v].acceleration;//时间间隔为1秒	
										if(positionNew<RoadNetworkRes[n].Road[r][l].BufferRoad[v-1].position)
										{
											positionNew=RoadNetworkRes[n].Road[r][l].BufferRoad[v].position;
										}
									}
									else
									{
										//根据GM模型更新加速度
										accelerationNew=1*RoadNetworkRes[n].Road[r][l].BufferRoad[v].speed*(0-RoadNetworkRes[n].Road[r][l].BufferRoad[v].speed)/(VehicleLength*RoadSIZE-RoadNetworkRes[n].Road[r][l].BufferRoad[v].position);//1为反映强度系数					

										//更新速度				
										speedNew=RoadNetworkRes[n].Road[r][l].BufferRoad[v].speed+RoadNetworkRes[n].Road[r][l].BufferRoad[v].acceleration*1;
										if(speedNew<0)
										{
											speedNew=0;
										}

										//更新位置
										positionNew=RoadNetworkRes[n].Road[r][l].BufferRoad[v].position+RoadNetworkRes[n].Road[r][l].BufferRoad[v].speed+0.5*RoadNetworkRes[n].Road[r][l].BufferRoad[v].acceleration;//时间间隔为1秒
										if(positionNew<RoadNetworkRes[n].Road[r][l].BufferRoad[v].position)
										{
											positionNew=RoadNetworkRes[n].Road[r][l].BufferRoad[v].position;
										}	

									}

								}
							}
							if (RoadNetworkRes[n].TrafficLight.GreenLightStatus_Phase4==true)
							{
								if ((r%2!=0 && l==0) || l==2)
								{
									if(RoadNetworkRes[n].Road[r][l].BufferRoad[v].speed<RoadNetworkRes[n].Road[r][l].BufferRoad[v].ExpSpeed)
										accelerationNew=RoadNetworkRes[n].Road[r][l].BufferRoad[v].acceleration+1;
									else
										accelerationNew=0;//速度达到期望速度后转入匀速行驶

									speedNew=RoadNetworkRes[n].Road[r][l].BufferRoad[v].speed+RoadNetworkRes[n].Road[r][l].BufferRoad[v].acceleration*1;
									if(speedNew<0)
									{
										speedNew=0;
									}
									positionNew=RoadNetworkRes[n].Road[r][l].BufferRoad[v].position+RoadNetworkRes[n].Road[r][l].BufferRoad[v].speed*1+0.5*RoadNetworkRes[n].Road[r][l].BufferRoad[v].acceleration;//时间间隔为1秒	
									if(positionNew<RoadNetworkRes[n].Road[r][l].BufferRoad[v-1].position)
									{
										positionNew=RoadNetworkRes[n].Road[r][l].BufferRoad[v].position;
									}
								}
								else
								{
									if(VehicleLength*RoadSIZE-RoadNetworkRes[n].Road[r][l].BufferRoad[v].position>150)//代表道路总长，起点为0，则道路总长为交通灯所在位置
									{
										if(RoadNetworkRes[n].Road[r][l].BufferRoad[v].speed<RoadNetworkRes[n].Road[r][l].BufferRoad[v].ExpSpeed)
											accelerationNew=RoadNetworkRes[n].Road[r][l].BufferRoad[v].acceleration+1;
										else
											accelerationNew=0;//速度达到期望速度后转入匀速行驶

										speedNew=RoadNetworkRes[n].Road[r][l].BufferRoad[v].speed+RoadNetworkRes[n].Road[r][l].BufferRoad[v].acceleration*1;
										if(speedNew<0)
										{
											speedNew=0;
										}
										positionNew=RoadNetworkRes[n].Road[r][l].BufferRoad[v].position+RoadNetworkRes[n].Road[r][l].BufferRoad[v].speed*1+0.5*RoadNetworkRes[n].Road[r][l].BufferRoad[v].acceleration;//时间间隔为1秒	
										if(positionNew<RoadNetworkRes[n].Road[r][l].BufferRoad[v-1].position)
										{
											positionNew=RoadNetworkRes[n].Road[r][l].BufferRoad[v].position;
										}
									}
									else
									{
										//根据GM模型更新加速度
										accelerationNew=1*RoadNetworkRes[n].Road[r][l].BufferRoad[v].speed*(0-RoadNetworkRes[n].Road[r][l].BufferRoad[v].speed)/(VehicleLength*RoadSIZE-RoadNetworkRes[n].Road[r][l].BufferRoad[v].position);//1为反映强度系数					

										//更新速度				
										speedNew=RoadNetworkRes[n].Road[r][l].BufferRoad[v].speed+RoadNetworkRes[n].Road[r][l].BufferRoad[v].acceleration*1;
										if(speedNew<0)
										{
											speedNew=0;
										}

										//更新位置
										positionNew=RoadNetworkRes[n].Road[r][l].BufferRoad[v].position+RoadNetworkRes[n].Road[r][l].BufferRoad[v].speed+0.5*RoadNetworkRes[n].Road[r][l].BufferRoad[v].acceleration;//时间间隔为1秒
										if(positionNew<RoadNetworkRes[n].Road[r][l].BufferRoad[v].position)
										{
											positionNew=RoadNetworkRes[n].Road[r][l].BufferRoad[v].position;
										}	

									}

								}
							}
						}

						//新数据赋值

						RoadNetworkRes[n].Road[r][l].BufferRoad[v].position=positionNew;
						RoadNetworkRes[n].Road[r][l].BufferRoad[v].speed=speedNew;
						RoadNetworkRes[n].Road[r][l].BufferRoad[v].acceleration=accelerationNew;		
					
						if (positionNew>0)
						{
							RoadNetworkRes[n].Road[r][l].BufferRoad[v].RoadTimeConsumed=RoadNetworkRes[n].Road[r][l].BufferRoad[v].RoadTimeConsumed+1;
						}
					}
				}
			}
		}
	}
}


int GetODList(int Roadnum, int Lanenum)
{
	int destinationRoadNum;
	int i,j;
	i=Roadnum;
	j=Lanenum;
	if (i==0)//北
	{
		if (j==0)//左转
		{
			destinationRoadNum=3;
		}
		if (j==1)//直行
		{
			destinationRoadNum=2;
		}
		if (j==2)//右转
		{
			destinationRoadNum=1;
		}
	}
	if (i==1)//西
	{
		if (j==0)
		{
			destinationRoadNum=0;
		}
		if (j==1)
		{
			destinationRoadNum=3;
		}
		if (j==2)
		{
			destinationRoadNum=2;
		}
	}
	if (i==2)//南
	{
		if (j==0)
		{
			destinationRoadNum=1;
		}
		if (j==1)
		{
			destinationRoadNum=0;
		}
		if (j==2)
		{
			destinationRoadNum=3;
		}
	}
	if (i==3)//东
	{
		if (j==0)
		{
			destinationRoadNum=2;
		}
		if (j==1)
		{
			destinationRoadNum=1;
		}
		if (j==2)
		{
			destinationRoadNum=0;
		}
	}
	return destinationRoadNum;
}

void DataPorcess()
//每个仿真时刻的仿真数据后处理
{
	int destinationRoad=-1;
	int NodeNumTemp=-1;
	int RoadNumTemp=-1;
	int LaneNumTemp=-1;
	VehicleAgent VehicleTemp;

	srand(int(time(0)));

	for (int n=0;n<NodeNumTol;n++)
	{
		for (int r=0;r<NumofRoads;r++)
		{
			for (int l=0;l<NumofLanes;l++)
			{

				destinationRoad=GetODList(r,l);//本节点内的目标路号
				if (RoadNetworkRes[n].NodeLinkTo[destinationRoad]==-1)
				//驶出路网 删除车辆
				{
					int front=RoadNetworkRes[n].Road[r][l].RoadFront;
					if (RoadNetworkRes[n].Road[r][l].BufferRoad[front].position>RoadSIZE*VehicleLength+1)
					{
						RoadNetworkRes[n].Road[r][l].BufferRoad[front].position=-1;
						RoadNetworkRes[n].Road[r][l].BufferRoad[front].speed=0;
						RoadNetworkRes[n].Road[r][l].BufferRoad[front].acceleration=0;
						RoadNetworkRes[n].Road[r][l].BufferRoad[front].VehicleFlag=false;
						RoadNetworkRes[n].Road[r][l].BufferRoad[front].ExpSpeed=0;

						RoadNetworkRes[n].Road[r][l].RoadFront=(RoadNetworkRes[n].Road[r][l].RoadFront+1)%RoadSIZE;
						RoadNetworkRes[n].Road[r][l].RoadCount--;

						RoadNetworkRes[n].RoadTraveTime[r]=RoadNetworkRes[n].RoadTraveTime[r]+RoadNetworkRes[n].Road[r][l].BufferRoad[front].RoadTimeConsumed;
						RoadNetworkRes[n].Road[r][l].BufferRoad[front].RoadTimeConsumed=0;

						RoadNetworkRes[n].VehicleInNumInAll[r]=RoadNetworkRes[n].VehicleInNumInAll[r]+1;

						RoadNetworkRes[n].RoadVehiclePassNum[r][l]=RoadNetworkRes[n].RoadVehiclePassNum[r][l]+1;
					}		
					
				}
				else
				//从本道路驶向其他道路
				{
					int front=RoadNetworkRes[n].Road[r][l].RoadFront;
					if (RoadNetworkRes[n].Road[r][l].BufferRoad[front].position>RoadSIZE*VehicleLength+1)
					{
						NodeNumTemp=RoadNetworkRes[n].NodeLinkTo[destinationRoad];//目标节点号
						RoadNumTemp=RoadNetworkRes[n].RoadLinkTo[destinationRoad];//目标节点的目标道路
						
						float temp=(float)rand()/RAND_MAX;
						if (temp<=0.5 && temp>=0)
						{
							LaneNumTemp=1;
						}
						if (temp>0.5 && temp<=0.8)
						{
							LaneNumTemp=0;
						}
						if (temp>0.8)
						{
							LaneNumTemp=2;
						}
						

						int front=RoadNetworkRes[n].Road[r][l].RoadFront;
						VehicleTemp=RoadNetworkRes[n].Road[r][l].BufferRoad[front];

						//原车辆数据删除
						RoadNetworkRes[n].Road[r][l].BufferRoad[front].position=-1;
						RoadNetworkRes[n].Road[r][l].BufferRoad[front].speed=0;
						RoadNetworkRes[n].Road[r][l].BufferRoad[front].acceleration=0;
						RoadNetworkRes[n].Road[r][l].BufferRoad[front].VehicleFlag=false;
						RoadNetworkRes[n].Road[r][l].BufferRoad[front].ExpSpeed=0;

						RoadNetworkRes[n].Road[r][l].RoadFront=(RoadNetworkRes[n].Road[r][l].RoadFront+1)%RoadSIZE;
						RoadNetworkRes[n].Road[r][l].RoadCount--;

						RoadNetworkRes[n].RoadTraveTime[r]=RoadNetworkRes[n].RoadTraveTime[r]+RoadNetworkRes[n].Road[r][l].BufferRoad[front].RoadTimeConsumed;
						RoadNetworkRes[n].Road[r][l].BufferRoad[front].RoadTimeConsumed=0;

						RoadNetworkRes[n].VehicleInNumInAll[r]=RoadNetworkRes[n].VehicleInNumInAll[r]+1;

						RoadNetworkRes[n].RoadVehiclePassNum[r][l]=RoadNetworkRes[n].RoadVehiclePassNum[r][l]+1;

						//插入新的目标道路(rear+1)%RoadSIZE!=front
						if ((RoadNetworkRes[NodeNumTemp].Road[RoadNumTemp][LaneNumTemp].RoadRear+1)%RoadSIZE!=RoadNetworkRes[NodeNumTemp].Road[RoadNumTemp][LaneNumTemp].RoadFront)
						{
							VehicleTemp.position=0;
							VehicleTemp.speed=VehicleTemp.speed/2;
							VehicleTemp.acceleration=0;
							VehicleTemp.RoadTimeConsumed=0;


							int rear=RoadNetworkRes[NodeNumTemp].Road[RoadNumTemp][LaneNumTemp].RoadRear;
							RoadNetworkRes[NodeNumTemp].Road[RoadNumTemp][LaneNumTemp].BufferRoad[rear].position=VehicleTemp.position;
							RoadNetworkRes[NodeNumTemp].Road[RoadNumTemp][LaneNumTemp].BufferRoad[rear].speed=VehicleTemp.speed;
							RoadNetworkRes[NodeNumTemp].Road[RoadNumTemp][LaneNumTemp].BufferRoad[rear].acceleration=VehicleTemp.acceleration;
							RoadNetworkRes[NodeNumTemp].Road[RoadNumTemp][LaneNumTemp].BufferRoad[rear].RoadTimeConsumed=VehicleTemp.RoadTimeConsumed;
							RoadNetworkRes[NodeNumTemp].Road[RoadNumTemp][LaneNumTemp].BufferRoad[rear].ExpSpeed=VehicleTemp.ExpSpeed;
							RoadNetworkRes[NodeNumTemp].Road[RoadNumTemp][LaneNumTemp].BufferRoad[rear].VehicleFlag=true;
							RoadNetworkRes[NodeNumTemp].Road[RoadNumTemp][LaneNumTemp].RoadRear=(RoadNetworkRes[NodeNumTemp].Road[RoadNumTemp][LaneNumTemp].RoadRear+1)%RoadSIZE;//循环队列的操作
							RoadNetworkRes[NodeNumTemp].Road[RoadNumTemp][LaneNumTemp].RoadCount++;
						}
						
					}
					
				}

			}
		}
	}
	
}

void DateProcessEnd()
//最终的数据处理，获得相关数据，计算路段关联度
{
	for (int n=0;n<NodeNumTol;n++)
	{
		for (int r=0;r<NumofRoads;r++)
		{
			/*//计算t
			RoadNetworkRes[n].RoadTraveTime[r]=(float)(RoadNetworkRes[n].RoadTraveTime[r]/(60*RoadNetworkRes[n].VehicleInNumInAll[r]));
			//计算Qmax
			int temp=-1;
			for (int l=0;l<NumofLanes;l++)
			{
				if (RoadNetworkRes[n].RoadVehiclePassNum[r][l]>temp)
				{
					RoadNetworkRes[n].VehicleInNumInMax[r]=RoadNetworkRes[n].RoadVehiclePassNum[r][l];
					temp=RoadNetworkRes[n].RoadVehiclePassNum[r][l];
				}
			}
			//计算Qtotal
			RoadNetworkRes[n].VehicleInNumInAll[r]=(int)(RoadNetworkRes[n].VehicleInNumInAll[r]/NumofLanes);
			float ratio=(float)RoadNetworkRes[n].VehicleInNumInMax[r]/RoadNetworkRes[n].VehicleInNumInAll[r];

			/////////计算各路段关联度
			RoadNetworkRes[n].RoadRelationNum[r]=(0.5/(1+RoadNetworkRes[n].RoadTraveTime[r]))*(ratio-1);
			//RoadNetworkRes[n].RoadRelationNum[r]=(ratio-(1/(1+RoadNetworkRes[n].RoadTraveTime[r])))/2;
			if (RoadNetworkRes[n].RoadRelationNum[r]>1)
			{
				RoadNetworkRes[n].RoadRelationNum[r]=1;
			}*/

			RoadNetworkRes[n].RoadRelationNum[r]=(float)RoadNetworkRes[n].VehicleInNumInAll[r]/(2*RoadSIZE*VehicleLength*NumofLanes);
			if (RoadNetworkRes[n].RoadRelationNum[r]>1)
			{
				RoadNetworkRes[n].RoadRelationNum[r]=1;
			}

		}
	}

	//填写laplace矩阵
	for (int n=0;n<NodeNumTol;n++)
	{
		for (int r=0;r<NumofRoads;r++)
		{
			int Dn=RoadNetworkRes[n].NodeLinkTo[r];
			
			if (Dn!=-1)
			{
				LapMatirx[n][Dn]=RoadNetworkRes[n].RoadRelationNum[r];
			}
		}
	}

	for (int i=0;i<NodeNumTol;i++)
	{
		for (int j=0;j<NodeNumTol;j++)
		{
			if (LapMatirx[i][j]>LapMatirx[j][i])
			{
				LapMatirx[j][i]=LapMatirx[i][j];
			}
			else
			{
				LapMatirx[i][j]=LapMatirx[j][i];
			}
		}
	}

	

	for (int n=0;n<NodeNumTol;n++)
	{
		float count=0;
		float temp=0;
		for(int k=0;k<NodeNumTol;k++)
		{
			if (LapMatirx[n][k]!=-1)
			{
				count=count+1;
				temp=temp+LapMatirx[n][k];
			}
		}
		Paint[n]=(float)temp/count;
	}


	//写画图数据
	FILE * PaintData;
	PaintData=fopen("PaintData.txt","w+");

	for (int i=0;i<NodeNumTol;i++)
	{	
		if (i>0 && i%RowSize==0)
		{
			fprintf(PaintData,"\n");

		}
		fprintf(PaintData,"%f ",Paint[i]);
	}
	
	fclose(PaintData);

	//写关联度数据文件
	FILE * RelationData;
	RelationData=fopen("RelationData.txt","w+");

	for (int n=0;n<NodeNumTol;n++)
	{	
		for (int k=0;k<NodeNumTol;k++)
		{
			if (LapMatirx[n][k]!=-1 && k>n)
			{	
				fprintf(RelationData,"%d--%d : %f \n",n,k,LapMatirx[n][k]);
			}	
		}
		
	}

	fclose(RelationData);


}


void PeopleInitial()
{
	
	srand(time(0));
	for (int i=0;i<PeopleNum;i++)
	{
		//编码
		PeopleCurrent[i].PeopleIndex=i;
		for (int j=0;j<RoadNum;j++)
		{
			PeopleCurrent[i].Chomosome[j]=rand()%2;
		}
		for (int ii=0;ii<NodeNumTol;ii++)
		{
			for (int jj=0;jj<NodeNumTol;jj++)
			{
				if (ii>=jj)
				{
					PeopleCurrent[i].LapMatrixBinary[ii][jj]=-1;
					PeopleNew[i].LapMatrixBinary[ii][jj]=-1;
				}
				else
				{
					PeopleCurrent[i].LapMatrixBinary[ii][jj]=0;
					PeopleNew[i].LapMatrixBinary[ii][jj]=0;
				}
			}
		}

		//划分方案对应的子区数目付初值
		PeopleCurrent[i].N=0;
		PeopleNew[i].N=0;
		//划分方案对应的子区适应度数组付初值
		for(int sa=0;sa<NodeNumTol;sa++)	
		{
			for (int sa2=0;sa2<NodeNumTol;sa2++)
			{
				PeopleCurrent[i].NRelation[sa][sa2]=-1;
				PeopleNew[i].NRelation[sa][sa2]=-1;
			}			
		}
		
		//划分方案适应度付初值
		PeopleCurrent[i].fitness=0;
		PeopleNew[i].fitness=0;


		//检验染色体合理性
		for (int j=0;j<RoadNum;j++)
		{
			if (LapVector[j]<0.2)
			{
				PeopleCurrent[i].Chomosome[j]=0;
			}
			if (LapVector[j]>0.8)
			{
				PeopleCurrent[i].Chomosome[j]=1;
			}
		}


	}

}


void EvaluatePeopleCurrent()
{
	for (int i=0;i<NodeNumTol;i++)
	{
		for (int j=0;j<NodeNumTol;j++)
		{
			LapMatirx2[i][j]=LapMatirx[i][j];
		}
	}

	//所有与划分方案评价相关的变量初始化
	for (int i=0;i<PeopleNum;i++)
	{
		//划分方案对应的子区数目付初值
		PeopleCurrent[i].N=0;
		//划分方案对应的子区适应度数组付初值
		for(int sa=0;sa<NodeNumTol;sa++)	
		{
			for (int sa2=0;sa2<NodeNumTol;sa2++)
			{
				PeopleCurrent[i].NRelation[sa][sa2]=-1;
			}

		}
		//划分方案适应度赋初值
		PeopleCurrent[i].fitness=0;
	}


	int TempIdx=0;
	//解码
	for (int i=0;i<PeopleNum;i++)
	{
		TempIdx=0;
		for (int ii=0;ii<NodeNumTol;ii++)
		{
			for (int jj=0;jj<NodeNumTol;jj++)
			{
				if (jj>ii && LapMatirx2[ii][jj]!=-1)
				{
					PeopleCurrent[i].LapMatrixBinary[ii][jj]=PeopleCurrent[i].Chomosome[TempIdx];
					TempIdx++;
				}
			}
		}
	}

	int a=0;

	//以PeopleCurrent[i].LapMatrixBinary[ii][jj]为依据判别子区数量以及区域关联度 从而得出方案的评价值
	bool CheckFlag=true;
	int roottemp=-1;
	int treeNumber;
	int root;

	int TreeIdx;
	//存储树节点的叶节点数据的一维数组索引变量
	int LeafIdx;
	//确定当前树的根节点			
	int rootIdx;

	for (int i=0;i<PeopleNum;i++)
	{
		for (int nf=0;nf<NodeNumTol;nf++)
		{
			NodeFlag[nf]=true;
			LeafFlag[nf]=true;
		}
		CheckFlag=true;


		treeNumber=0;
		root=100000;
		
		while (root!=-1)//如果还能生成树就继续
		{
			//控制子区的划分数目
			//PeopleCurrent[i].N=PeopleCurrent[i].N+1;

			//一次循环生成一棵树 存储结构为一维数组tree

			for (int mm=0;mm<NodeNumTol;mm++)
			{
				tree[mm].NodeIndex=-1;
				for (int mml=0;mml<4;mml++)
				{
					tree[mm].Leaf[mml]=-1;
				}
				
			}
			
			for(int Tidx=0;Tidx<NodeNumTol;Tidx++)
			{
				for (int Lidx=0;Lidx<4;Lidx++)
				{
					tree[Tidx].NodeIndex=-1;
					tree[Tidx].Leaf[Lidx]=-1;
				}
			}


			//存储树所有节点数据的一维数组索引变量
			 TreeIdx=1;
			//存储树节点的叶节点数据的一维数组索引变量
			 LeafIdx=0;
			//确定当前树的根节点			
			rootIdx=0;
			root=-1;
			for (int k=0;k<NodeNumTol;k++)
			{
				if (NodeFlag[k]==true)
				{
					root=k;
					break;
				}
			}
			tree[rootIdx].NodeIndex=root;
			roottemp=root;
			
			// 寻找到本树的所有节点
			while (rootIdx<TreeIdx)
			{
				// 根节点对应的叶节点序号赋值
				for (int k=0;k<NodeNumTol;k++)
				{
					if (PeopleCurrent[i].LapMatrixBinary[roottemp][k]==1 && NodeFlag[k]==true)
					{	
						tree[rootIdx].Leaf[LeafIdx]=k;
						LeafIdx++;
					}
				}
				//找到当前root所有的leaf之后将root标志状态更改
				NodeFlag[roottemp]=false;
				
				//将root的leaf节点加入tree数组
				int leaftemp=-1;
				for (int k=0;k<4;k++)
				{
					if (tree[rootIdx].Leaf[k]!=-1 && LeafFlag[tree[rootIdx].Leaf[k]]==true)
					{
						leaftemp=tree[rootIdx].Leaf[k];
						
						tree[TreeIdx].NodeIndex=leaftemp;
						TreeIdx++;

						LeafFlag[leaftemp]=false;
					}
				}

				//重定位root
				rootIdx++;
				if (rootIdx==TreeIdx)
				{
					NodeFlag[tree[TreeIdx].NodeIndex]=false;
				}

				if (tree[rootIdx].NodeIndex!=-1)
				{
					roottemp=tree[rootIdx].NodeIndex;
				}				
			}
			

			
			if (root!=-1)
			{
				for (int leafCount=0;leafCount<NodeNumTol;leafCount++)
				{
					if (tree[leafCount].NodeIndex!=-1)
					{
						PeopleCurrent[i].NRelation[treeNumber][leafCount]=tree[leafCount].NodeIndex;
					}
				}

				treeNumber=treeNumber+1;
			}
		}


		//计算不同划分方案评分
		PeopleCurrent[i].N=treeNumber;


		int AreaNumber=treeNumber;
		float AreaScore=0;
		
		for (int tempindex=0;tempindex<NodeNumTol;tempindex++)
		{
			temp[tempindex]=temp2[tempindex]=temp3[tempindex]=-1;
		}


		int temp2index=0;
		for (int nn=0;nn<AreaNumber;nn++)
		{
			
			for (int n2=0;n2<NodeNumTol;n2++)
			{
				temp[n2]=PeopleCurrent[i].NRelation[nn][n2];
			}

			//对temp进行赋值并排序
			temp2index=0;
			for (int n3=0;n3<NodeNumTol;n3++)
			{
				int x=temp[n3];
				int y=-1;
				for (int n4=n3+1;n4<NodeNumTol;n4++)
				{
					y=temp[n4];
					if (x!=-1 && y!=-1)
					{
						if (LapMatirx2[x][y]!=-1)
						{
							temp2[temp2index]=LapMatirx2[x][y];
							temp2index++;
						}
						
					}
					
				}	
			}
			//排序
			for (int a=0;a<NodeNumTol;a++)
			{
				float mark=10000;
				int markIdx=-1;
				for (int a1=0;a1<NodeNumTol;a1++)
				{
					if (temp2[a1]<mark  &&  temp2[a1]>0)
					{
						mark=temp2[a1];
						markIdx=a1;
					}
				}
				temp3[a]=mark;
				temp2[markIdx]=-1;
			}

			//计算子区域联合关联度
			float SubareaScore=1;
			for (int a=0;a<NodeNumTol;a++)
			{
				
				if (temp3[a]!=-1 && temp3[a]!=10000)
				{
					if (a==0)
					{
						SubareaScore=SubareaScore*pow(temp3[a],1);
					}
					else
					{
						SubareaScore=SubareaScore*pow(temp3[a],1/a);
					}
					
				}
			}
			
			if (SubareaScore==1 || SubareaScore==10000)
			{
				SubareaScore=0;
			}

			AreaScore=AreaScore+SubareaScore;
			
		}

		PeopleCurrent[i].fitness=-pow(PeopleCurrent[i].N,2.0)+AreaScore;
	}

	int aa=0;

}

void CrossoverAndMutationOperation()
{
	srand(time(0));
	//交叉
	for (int i=0;i<PeopleNum;i++)
	{
		float pc=(float)rand()/RAND_MAX;
		if (pc<PCross)
		{
			int anotherPlan=rand()%PeopleNum;
			int crosspoint=rand()%RoadNum;
			
			for (int j=0;j<crosspoint;j++)
			{
				PeopleNew[i].Chomosome[j]=PeopleCurrent[i].Chomosome[j];
			}
			for (int j=crosspoint;j<RoadNum;j++)
			{
				PeopleNew[i].Chomosome[j]=PeopleCurrent[anotherPlan].Chomosome[j];
			}
			
		}
	}
	//变异
	for (int i=0;i<PeopleNum;i++)
	{
		for (int j=0;j<RoadNum;j++)
		{
			float pm=(float)rand()/RAND_MAX;
			if (pm<PMutate)
			{
				if (PeopleNew[i].Chomosome[j]==0)
				{
					PeopleNew[i].Chomosome[j]=1;
				}
				else
				{
					PeopleNew[i].Chomosome[j]=0;
				}
			}
			
		}
		
	}

	for (int i=0;i<PeopleNum;i++)
	{
		//检验染色体合理性
		for (int j=0;j<RoadNum;j++)
		{
			if (LapVector[j]<0.2)
			{
				PeopleNew[i].Chomosome[j]=0;
			}
			if (LapVector[j]>0.8)
			{
				PeopleNew[i].Chomosome[j]=1;
			}
		}

	}

	

}

void EvaluatePeopleNew()
{
	/*for (int i=0;i<NodeNumTol;i++)
	{
		for (int j=0;j<NodeNumTol;j++)
		{
			LapMatirx2[i][j]=LapMatirx[i][j];
		}
	}*/

	//所有与划分方案评价相关的变量初始化
	for (int i=0;i<PeopleNum;i++)
	{
		//划分方案对应的子区数目赋初值
		PeopleNew[i].N=0;
		//划分方案对应的子区适应度数组赋初值
		for(int sa=0;sa<NodeNumTol;sa++)	
		{
			for (int sa2=0;sa2<NodeNumTol;sa2++)
			{
				PeopleNew[i].NRelation[sa][sa2]=-1;
			}
		}
		//划分方案适应度赋初值
		PeopleNew[i].fitness=0;
	}


	int TempIdx=0;
	//解码
	for (int i=0;i<PeopleNum;i++)
	{
		TempIdx=0;
		for (int ii=0;ii<NodeNumTol;ii++)
		{
			for (int jj=0;jj<NodeNumTol;jj++)
			{
				if (jj>ii && LapMatirx2[ii][jj]!=-1)
				{
					PeopleNew[i].LapMatrixBinary[ii][jj]=PeopleNew[i].Chomosome[TempIdx];
					TempIdx++;
				}
			}
		}
	}

	int a=0;

	//以PeopleCurrent[i].LapMatrixBinary[ii][jj]为依据判别子区数量以及区域关联度 从而得出方案的评价值
	for (int NodeFlagIdx=0;NodeFlagIdx<NodeNumTol;NodeFlagIdx++)
	{
		NodeFlag[NodeFlagIdx]=true;
	}

	bool CheckFlag=true;
	int roottemp=-1;
	int root;
	int treeNumber;

	//存储树所有节点数据的一维数组索引变量
	int TreeIdx;
	//存储树节点的叶节点数据的一维数组索引变量
	int LeafIdx;
	//确定当前树的根节点			
	int rootIdx;

	for (int i=0;i<PeopleNum;i++)
	{
		for (int nf=0;nf<NodeNumTol;nf++)
		{
			NodeFlag[nf]=true;
			LeafFlag[nf]=true;
		}
		CheckFlag=true;


		treeNumber=0;
		root=100000;
		
		while (root!=-1)//如果还能生成树就继续
		{
			//控制子区的划分数目
			//PeopleCurrent[i].N=PeopleCurrent[i].N+1;

			//一次循环生成一棵树 存储结构为一维数组tree
			for (int mm=0;mm<NodeNumTol;mm++)
			{
				tree[mm].NodeIndex=-1;
				for (int mml=0;mml<4;mml++)
				{
					tree[mm].Leaf[mml]=-1;
				}
			}

			for(int Tidx=0;Tidx<NodeNumTol;Tidx++)
			{
				for (int Lidx=0;Lidx<4;Lidx++)
				{
					tree[Tidx].NodeIndex=-1;
					tree[Tidx].Leaf[Lidx]=-1;
				}
			}


			//存储树所有节点数据的一维数组索引变量
			TreeIdx=1;
			//存储树节点的叶节点数据的一维数组索引变量
			LeafIdx=0;
			//确定当前树的根节点			
			rootIdx=0;
			root=-1;
			for (int k=0;k<NodeNumTol;k++)
			{
				if (NodeFlag[k]==true)
				{
					root=k;
					break;
				}
			}
			tree[rootIdx].NodeIndex=root;
			roottemp=root;

			// 寻找到本树的所有节点
			while (rootIdx<TreeIdx)
			{
				// 根节点对应的叶节点序号赋值
				for (int k=0;k<NodeNumTol;k++)
				{
					if (PeopleNew[i].LapMatrixBinary[roottemp][k]==1 && NodeFlag[k]==true)
					{	
						tree[rootIdx].Leaf[LeafIdx]=k;
						LeafIdx++;
					}
				}
				//找到当前root所有的leaf之后将root标志状态更改
				NodeFlag[roottemp]=false;

				//将root的leaf节点加入tree数组
				int leaftemp=-1;
				for (int k=0;k<4;k++)
				{
					if (tree[rootIdx].Leaf[k]!=-1 && LeafFlag[tree[rootIdx].Leaf[k]]==true)
					{
						leaftemp=tree[rootIdx].Leaf[k];

						tree[TreeIdx].NodeIndex=leaftemp;
						TreeIdx++;

						LeafFlag[leaftemp]=false;
					}
				}

				//重定位root
				rootIdx++;
				if (rootIdx==TreeIdx)
				{
					NodeFlag[tree[TreeIdx].NodeIndex]=false;
				}

				if (tree[rootIdx].NodeIndex!=-1)
				{
					roottemp=tree[rootIdx].NodeIndex;
				}
			}



			if (root!=-1)
			{
				for (int leafCount=0;leafCount<NodeNumTol;leafCount++)
				{
					if (tree[leafCount].NodeIndex!=-1)
					{
						PeopleNew[i].NRelation[treeNumber][leafCount]=tree[leafCount].NodeIndex;
					}
				}

				treeNumber=treeNumber+1;
			}
		}


		//计算不同划分方案评分
		PeopleNew[i].N=treeNumber;

		int AreaNumber=treeNumber;
		float AreaScore=0;
		
		for (int tempindex=0;tempindex<NodeNumTol;tempindex++)
		{
			temp[tempindex]=temp2[tempindex]=temp3[tempindex]=-1;
		}


		int temp2index=0;
		for (int nn=0;nn<AreaNumber;nn++)
		{
			for (int n2=0;n2<NodeNumTol;n2++)
			{
				temp[n2]=PeopleNew[i].NRelation[nn][n2];
			}

			//对temp进行赋值并排序
			temp2index=0;
			for (int n3=0;n3<NodeNumTol;n3++)
			{
				int x=temp[n3];
				int y=-1;
				for (int n4=n3+1;n4<NodeNumTol;n4++)
				{
					y=temp[n4];
					if (x!=-1 && y!=-1 )
					{
						if (LapMatirx2[x][y]!=-1)
						{
							temp2[temp2index]=LapMatirx2[x][y];
							temp2index++;
						}
						
					}

				}	
			}
			//排序
			for (int a=0;a<NodeNumTol;a++)
			{
				float mark=10000;
				int markIdx=-1;
				for (int a1=0;a1<NodeNumTol;a1++)
				{
					if (temp2[a1]<mark  &&  temp2[a1]>0)
					{
						mark=temp2[a1];
						markIdx=a1;
					}
				}
				temp3[a]=mark;
				temp2[markIdx]=-1;
			}

			//计算子区域联合关联度
			float SubareaScore=1;
			for (int a=0;a<NodeNumTol;a++)
			{

				if (temp3[a]!=-1 && temp3[a]!=10000)
				{
					if (a==0)
					{
						SubareaScore=SubareaScore*pow(temp3[a],1);
					}
					else
					{
						SubareaScore=SubareaScore*pow(temp3[a],1/a);
					}

				}
			}

			if (SubareaScore==1 || SubareaScore==10000)
			{
				SubareaScore=0;
			}

			AreaScore=AreaScore+SubareaScore;

		}


		PeopleNew[i].fitness=-pow(PeopleNew[i].N,2.0)+AreaScore;

	}

}

void SelectionOpration()
{
	for (int i=0;i<PeopleNum;i++)
	{
		if (PeopleNew[i].fitness>PeopleCurrent[i].fitness)
		{
			PeopleCurrent[i]=PeopleNew[i];
		}

	}
}


int main(int argc, char* argv[])
{
///////////////////////////////////////////////////////////
//一、通过仿真计算路网中各路段关联度
	

	int ComputeMode;
	//0：CPU计算模式  其他：GPU计算模式
	ComputeMode=1;

	//初始化路网拓扑
	RoadNetworkInitial();

	//交通灯数据初始化
	TrafficLightsDataInitial();
	//路网数据初始化
	RoadNetworkDataInitial();


	//生成交通控制方案
	//TrafficControlProGen();
	TrafficControlProGen2();

	SimulationClock=0;
	SimulationDuration=10800;

	//FILE * recordTime;
	//recordTime=fopen("Count.txt","w+");

	while (SimulationClock<SimulationDuration)
	{
		//1、刷新交通灯状态
		TrafficLightsRefresh();	
		//2、按照车辆的出发时间将车辆插入到路网中 
		VehicleGeneration();
		//3、主体仿真函数--根据“跟车”和“换道”原则更新路网中车辆状态
		SimulationProcess();
		//4、数据后处理--车辆数据位置的改变
		DataPorcess();

		//fprintf(recordTime,"%d \n",RoadNetworkRes[7].Road[2][1].RoadCount);

		SimulationClock++;
		//在屏幕上显示当前的仿真时钟
		HANDLE hOut=GetStdHandle(STD_OUTPUT_HANDLE);
		COORD pos0;
		pos0.X=0;
		pos0.Y=0;
		SetConsoleCursorPosition(hOut,pos0);
		fprintf(stdout,"Simulation: CurrentTimestep is %d",SimulationClock);
	}

	
	//fclose(recordTime);

	clock_t start,end;
	float duration;
	start=clock();

	//最终的数据处理，获得相关数据，计算路段关联度
	DateProcessEnd();
	
///////////////////////////////////////////////////////////
//二、开始应用GA进行控制子区划分
	int TempIdx=0;
	for (int i=0;i<NodeNumTol;i++)
	{
		for (int j=0;j<NodeNumTol;j++)
		{
			if (LapMatirx[i][j]!=-1  && j>i)//从上三角矩阵中取值
			{
				LapVector[TempIdx]=LapMatirx[i][j];
				TempIdx++;
			}
		}
	}

	printf("\n");
	printf("Begin to divide the control area into subareas...\n");
	int iteration=0;

	//生成初始种群 P
	PeopleInitial();
	//计算种群 P 适应度 
	EvaluatePeopleCurrent();

	FILE * DivideIteration;
	DivideIteration=fopen("DivideIteration.txt","w+");

	while(iteration<MaxGen)
	{
		//交叉变异生成 W
		CrossoverAndMutationOperation();
		//评价种群 W 适应度
		EvaluatePeopleNew();
		//选择操作 更新 P 中的划分方案个体
		SelectionOpration();

		iteration++;

		HANDLE hOut=GetStdHandle(STD_OUTPUT_HANDLE);
		COORD pos0;
		pos0.X=0;
		pos0.Y=2;
		SetConsoleCursorPosition(hOut,pos0);
		fprintf(stdout,"Division: CurrentIteration is %d",iteration);

		//写文件
		float temp=-1000000;
		int tempmark=-1;
		for(int m=0;m<PeopleNum;m++)
		{
			if (PeopleCurrent[m].fitness>temp)
			{
				temp=PeopleCurrent[m].fitness;
				tempmark=m;
			}
		}
		fprintf(DivideIteration,"%f \n",PeopleCurrent[tempmark].fitness);
	}
	fclose(DivideIteration);

	
	//打印最佳划分方案
	float temp=-1000000;
	int tempmark=-1;
	for(int m=0;m<PeopleNum;m++)
	{
		if (PeopleCurrent[m].fitness>temp)
		{
			temp=PeopleCurrent[m].fitness;
			tempmark=m;
		}
	}

	FILE * Bestdivide;
	Bestdivide=fopen("Bestdivide.txt","w+");
	/*for (int m=0;m<RoadNum;m++)
	{
		fprintf(Bestdivide,"%d ",PeopleCurrent[tempmark].Chomosome[m]);
	}*/

	int TempIdx2=0;
	for (int ii=0;ii<NodeNumTol;ii++)
	{
		//TempIdx2=0;
		for (int jj=0;jj<NodeNumTol;jj++)
		{
			if (jj>ii && LapMatirx[ii][jj]!=-1 )
			{
				PeopleCurrent[tempmark].LapMatrixBinary[ii][jj]=PeopleCurrent[tempmark].Chomosome[TempIdx2];
				TempIdx2++;
			}
		}
	}
	for (int n=0;n<NodeNumTol;n++)
	{	
		for (int k=0;k<NodeNumTol;k++)
		{
			if (PeopleCurrent[tempmark].LapMatrixBinary[n][k]==1 && k>n)
			{	
				fprintf(Bestdivide,"%d--%d : %d \n",n,k,PeopleCurrent[tempmark].LapMatrixBinary[n][k]);
			}
		}
	}
	
	
	fclose(Bestdivide);

	end=clock();
	duration=end-start;

	printf("\nDivide Complete.");

	scanf("hold...");
	return 0;
}
