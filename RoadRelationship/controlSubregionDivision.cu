#include <stdio.h>
#include <stdlib.h>
// #include <cutil_inline.h>
#include <helper_functions.h>



//�꼰ȫ�ֱ�������
//////////////////////////////////////////////////////////////////////////
#define RowSize 5 //����5*5��lattice·��
#define NodeNumTol 25
// Do not consider the road beyond the lattice
#define RoadNum 40
#define NodeConnectNumMax 4

#define NumofRoads 4
// lanes for a road
#define NumofLanes 3

#define RoadSIZE 130 //��ʻ��·���������Ŀ
// in meter
#define VehicleLength 4

#define pi 3.1415926


#define NumofPhase 4  //ÿ��·����4���źŵƿ�����
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
	//�������
	// smaller number is in front
	int VehicleID;

	float position;//λ��
	float speed;//�ٶ�
	float acceleration;//���ٶ�
	bool VehicleFlag;//��ʾ����Ԫ�����Ƿ��г�
	float ExpSpeed;//�����ٶ�

	//�����ڵ���·����ʻ��ʱ
	int RoadTimeConsumed;
};

struct RoadAgent
{
	//��·���еĶ���
	VehicleAgent BufferRoad[RoadSIZE];//��������

	int RoadFront;//ͷָ��
	int RoadRear; //βָ��
	int RoadCount;//�����е�Ԫ�ظ���

};

struct TrafficLightsAgent
{
	//������̵�״̬
	// west and east go, then left, then south and north go, then left
	bool GreenLightStatus_Phase1;
	bool GreenLightStatus_Phase2;
	bool GreenLightStatus_Phase3;
	bool GreenLightStatus_Phase4;

	int cycle;//�źŵ�����

	int Phase;//���׼�źŵƵ���λ��

	//������̵Ƴ���ʱ��
	int GreenDuration__Phase1;
	int GreenDuration__Phase2;
	int GreenDuration__Phase3;
	int GreenDuration__Phase4;	
};

//��������·�����˵����ݽṹ
struct RoadNetworkResTemplate
{
	int NodeIndex;

	//���屾�ڵ��������·�������ڵ������Ľڵ����
	// 0, north, 1 west, 2 south, 3 east
	int NodeLinkTo[NumofRoads];
	// 0, south, 1 east, 2 north, 3 west
	int RoadLinkTo[NumofRoads];

	//���彻ͨ��
	TrafficLightsAgent TrafficLight;


	//�������ڼ���·�ι����ȵı���//////////////////////////////////
	//�������ݾ�Ϊ���복������
	float VehicleArrRate[NumofRoads];

	int RoadLength[NumofRoads];//ͨ�򱾽ڵ��·��·��	

	float RoadTraveTime[NumofRoads];//ͨ�򱾽ڵ��·�ϵĳ�������ʻʱ��
	
	int  VehicleInNumInAll[NumofRoads];//���򱾽ڵ�ĳ�������
	int RoadVehiclePassNum[NumofRoads][NumofLanes];//·��ͨ���ĳ�����Ŀ
	int  VehicleInNumInMax[NumofRoads];//���򱾽ڵ���������

	//�洢·������������
	float RoadRelationNum[NumofRoads];
	////////////////////////////////////////////////////////////////

	//������·�����ӵ�4����·
	RoadAgent Road[NumofRoads][NumofLanes];

};

RoadNetworkResTemplate RoadNetworkRes[NodeNumTol];
RoadNetworkResTemplate RoadNetworkResNew[NodeNumTol];


int SimulationClock;//�������ʱ��(s)
int SimulationDuration;//����������ʱ��(s)

//���ڵ������Ⱦ���
float LapMatirx[NodeNumTol][NodeNumTol];
float LapMatirx2[NodeNumTol][NodeNumTol];
float LapVector[RoadNum];

//��ͼ�õ����ݾ���
float Paint[NodeNumTol];

//�Ŵ��㷨�õ���Ⱦɫ�����
struct People
{
	int PeopleIndex;
	int Chomosome[RoadNum];
	int LapMatrixBinary[NodeNumTol][NodeNumTol];

	//������Ŀ
	int N;
	//�������������������--��໮��Ϊ25������ ��һ��·��һ������ ��ȫ����
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
//����·������ ��ȷ���ӹ�ϵ
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
			//��Ӧ�����һ��Ԫ��
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
			//��Ӧ�������һ��Ԫ��
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
			//��Ӧ�����һ��Ԫ��
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
			//��Ӧ�������һ��Ԫ��
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
			//�������Ե�������Ԫ��
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
//��ͨ�����ݳ�ʼ��
{
	for (int l=0;l<NodeNumTol;l++)
	{
		//���ཻͨ�Ƴ�ʼ��
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
//·�����ݳ�ʼ��
{
	for (int i=0;i<NodeNumTol;i++)
	{
		//if (i==12)
		{
			//Ϊ����·��������ֵ
			RoadNetworkRes[i].VehicleArrRate[0]=flow;
			
			RoadNetworkRes[i].VehicleArrRate[1]=flow;
		
			RoadNetworkRes[i].VehicleArrRate[2]=flow;
			
			RoadNetworkRes[i].VehicleArrRate[3]=flow;
			
		}
		//��һ��������--·�����ĵ�Ϊ������
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
		//�ڶ���������--·������Ϊ������
		
		
		//����·���ȸ�ֵ
		RoadNetworkRes[i].RoadLength[0]=RoadSIZE*VehicleLength;
		RoadNetworkRes[i].RoadLength[1]=RoadSIZE*VehicleLength;
		RoadNetworkRes[i].RoadLength[2]=RoadSIZE*VehicleLength;
		RoadNetworkRes[i].RoadLength[3]=RoadSIZE*VehicleLength;
		
		//����·ͨ���ĳ�����
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

		//����·����������ʻʱ��RoadTraveTime
		RoadNetworkRes[i].RoadTraveTime[0]=0;
		RoadNetworkRes[i].RoadTraveTime[1]=0;
		RoadNetworkRes[i].RoadTraveTime[2]=0;
		RoadNetworkRes[i].RoadTraveTime[3]=0;

		//����·���복����ֵ
		RoadNetworkRes[i].VehicleInNumInAll[0]=0;
		RoadNetworkRes[i].VehicleInNumInAll[1]=0;
		RoadNetworkRes[i].VehicleInNumInAll[2]=0;
		RoadNetworkRes[i].VehicleInNumInAll[3]=0;

		//����·�����������ֵ
		RoadNetworkRes[i].VehicleInNumInMax[0]=0;
		RoadNetworkRes[i].VehicleInNumInMax[1]=0;
		RoadNetworkRes[i].VehicleInNumInMax[2]=0;
		RoadNetworkRes[i].VehicleInNumInMax[3]=0;

		//����·�Ĺ���������
		RoadNetworkRes[i].RoadRelationNum[0]=0;
		RoadNetworkRes[i].RoadRelationNum[1]=0;
		RoadNetworkRes[i].RoadRelationNum[2]=0;
		RoadNetworkRes[i].RoadRelationNum[3]=0;
	}


	//��·�������ݸ�ֵ
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

void TrafficControlProGen()  //���ɽ�ͨ�ƿ��Ʒ���
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

		

		//��һ���̵Ƴ���ʱ��
		RoadNetworkRes[i].TrafficLight.GreenDuration__Phase1=GreenTimeInPhaseMin+((GreenTimeInPhaseMax-GreenTimeInPhaseMin)*((float)rand()/RAND_MAX));
		
		//�ڶ����̵Ƴ���ʱ��
		int temp = RoadNetworkRes[i].TrafficLight.cycle - RoadNetworkRes[i].TrafficLight.GreenDuration__Phase1;
		if (temp > GreenTimeInPhaseMax)
		{
			RoadNetworkRes[i].TrafficLight.GreenDuration__Phase2 = GreenTimeInPhaseMin + ((GreenTimeInPhaseMax-GreenTimeInPhaseMin) * ((float)rand()/RAND_MAX));
		}
		else
		{
			RoadNetworkRes[i].TrafficLight.GreenDuration__Phase2 = GreenTimeInPhaseMin + ((temp - GreenTimeInPhaseMin)*((float)rand()/RAND_MAX));
		}
		
		//�������̵Ƴ���ʱ��
		temp = RoadNetworkRes[i].TrafficLight.cycle-RoadNetworkRes[i].TrafficLight.GreenDuration__Phase1-RoadNetworkRes[i].TrafficLight.GreenDuration__Phase2;
		if (temp>GreenTimeInPhaseMax)
		{
			RoadNetworkRes[i].TrafficLight.GreenDuration__Phase3=GreenTimeInPhaseMin+((GreenTimeInPhaseMax-GreenTimeInPhaseMin)*((float)rand()/RAND_MAX));
		}
		else
		{
			RoadNetworkRes[i].TrafficLight.GreenDuration__Phase3=GreenTimeInPhaseMin+((temp-GreenTimeInPhaseMin)*((float)rand()/RAND_MAX));
		}

		//�������̵Ƴ���ʱ��
		int SumThree=RoadNetworkRes[i].TrafficLight.GreenDuration__Phase1+RoadNetworkRes[i].TrafficLight.GreenDuration__Phase2+RoadNetworkRes[i].TrafficLight.GreenDuration__Phase3;
		RoadNetworkRes[i].TrafficLight.GreenDuration__Phase4=RoadNetworkRes[i].TrafficLight.cycle - SumThree;
	}
}

void TrafficLightsRefresh()
//��ͨ���ź�ˢ��
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

int VehGenEvent[NodeNumTol][NumofRoads][NumofLanes];//�������������¼���ʱ���
double interval[NodeNumTol][NumofRoads][NumofLanes];
int intervalReal[NodeNumTol][NumofRoads][NumofLanes];

float RandomGen(float Lambda)
{
	float Z;
	float output;

	do 
	{
		Z=((float)rand()/RAND_MAX);//ZΪ0-1֮����ȷֲ������
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
			u1=((float)rand()/RAND_MAX);//u1Ϊ0-1֮����ȷֲ������
			u2=((float)rand()/RAND_MAX);//u2Ϊ0-1֮����ȷֲ������
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
//��������
{
	srand(int(time(0)));//�������������
	float randn;//�����С
	randn=rand();

	for (int n=0;n<NodeNumTol;n++)
	{
		for (int r=0;r<NumofRoads;r++)
		{
			for (int l=0;l<NumofLanes;l++)
			{
				if (SimulationClock==0)//�趨���濪ʼʱ����ĿΪ��
				{
					srand(unsigned(randn));
					randn=rand();

					float VehArriveRate=RoadNetworkRes[n].VehicleArrRate[r];//������������ͬһ������������

					interval[n][r][l]=RandomGen(VehArriveRate);	
					intervalReal[n][r][l]=int(interval[n][r][l]+1);//ת��Ϊ��������
					VehGenEvent[n][r][l]=SimulationClock+intervalReal[n][r][l];//������һ���������ʱ��

				}
				else
				{
					if (VehGenEvent[n][r][l]==SimulationClock)//�������������¼�
					{
						//****(1) ��������Ҫʵ�����������󣬲������·����
						srand(unsigned(randn));
						randn=rand();
						//���ɳ���
						VehicleAgent AgentCar;
						AgentCar.position=0;//·���ܳ���Ϊ1000�ף���ʼ����λ��Ϊ0��ע��Ҫ�����ڳ�ʼ��λ��-1
						AgentCar.speed=10;//float(abs(GaussRandomGen(40/3.6,0.0001)));//�ٶ�ƽ��ֵΪ40km/Сʱ
						AgentCar.acceleration=0;//�趨��ʼ���ٶ�Ϊ0�������ٽ��룬����Ҳ���Է��ϸ�˹�ֲ�����				
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
								RoadNetworkRes[n].Road[r][l].RoadRear=(RoadNetworkRes[n].Road[r][l].RoadRear+1)%RoadSIZE;//ѭ�����еĲ���
								RoadNetworkRes[n].Road[r][l].RoadCount++;
							}

						}
						//****(2) ������һ�γ��������ʱ��
						float VehArriveRate=RoadNetworkRes[n].VehicleArrRate[r];//������������ͬһ������������
						interval[n][r][l]=RandomGen(VehArriveRate);
						intervalReal[n][r][l]=int(interval[n][r][l]+1);//ת��Ϊ��������
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
//�������庯��
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


						//����ͷ��״̬����1
						if(v>0 && RoadNetworkRes[n].Road[r][l].BufferRoad[v-1].VehicleFlag==true)
						{
							if (RoadNetworkRes[n].Road[r][l].BufferRoad[v-1].position<RoadNetworkRes[n].Road[r][l].BufferRoad[v].position) //��������
							{
								positionNew=RoadNetworkRes[n].Road[r][l].BufferRoad[v].position;
								speedNew=0;
								accelerationNew=0;
							}
							if (RoadNetworkRes[n].Road[r][l].BufferRoad[v-1].position-RoadNetworkRes[n].Road[r][l].BufferRoad[v].position<=150 && RoadNetworkRes[n].Road[r][l].BufferRoad[v-1].position-RoadNetworkRes[n].Road[r][l].BufferRoad[v].position>0)
								//����״̬
							{
								if (RoadNetworkRes[n].Road[r][l].BufferRoad[v].position==0)//�շ����ĳ���
								{
									if (RoadNetworkRes[n].Road[r][l].BufferRoad[v-1].position>0) //ǰ���Ѵ�������״̬�����η�����Ҫ�����ʼ�ٶȹ���
									{
										while(RoadNetworkRes[n].Road[r][l].BufferRoad[v].speed*1>RoadNetworkRes[n].Road[r][l].BufferRoad[v-1].position)
											//���������ٶ�
										{
											RoadNetworkRes[n].Road[r][l].BufferRoad[v].speed=RoadNetworkRes[n].Road[r][l].BufferRoad[v].speed/2;
										}
										//���¼��ٶ�
										if(RoadNetworkRes[n].Road[r][l].BufferRoad[v].speed==0 && RoadNetworkRes[n].Road[r][l].BufferRoad[v-1].position-RoadNetworkRes[n].Road[r][l].BufferRoad[v].position>5)
											accelerationNew=RoadNetworkRes[n].Road[r][l].BufferRoad[v].acceleration+1;
										else
											accelerationNew=RoadNetworkRes[n].Road[r][l].BufferRoad[v].speed*(RoadNetworkRes[n].Road[r][l].BufferRoad[v-1].speed-RoadNetworkRes[n].Road[r][l].BufferRoad[v].speed)/(RoadNetworkRes[n].Road[r][l].BufferRoad[v-1].position-RoadNetworkRes[n].Road[r][l].BufferRoad[v].position);

										//�����ٶ�
										speedNew=RoadNetworkRes[n].Road[r][l].BufferRoad[v].speed+RoadNetworkRes[n].Road[r][l].BufferRoad[v].acceleration;
										if (speedNew<0)
										{
											speedNew=0;
											accelerationNew=0;
										}

										//����λ��
										positionNew=RoadNetworkRes[n].Road[r][l].BufferRoad[v].position+RoadNetworkRes[n].Road[r][l].BufferRoad[v].speed+0.5*RoadNetworkRes[n].Road[r][l].BufferRoad[v].acceleration;
										if (positionNew<RoadNetworkRes[n].Road[r][l].BufferRoad[v].position || positionNew>=RoadNetworkRes[n].Road[r][l].BufferRoad[v-1].position+RoadNetworkRes[n].Road[r][l].BufferRoad[v-1].speed+0.5*RoadNetworkRes[n].Road[r][l].BufferRoad[v-1].acceleration*RoadNetworkRes[n].Road[r][l].BufferRoad[v-1].acceleration)
										{
											positionNew=RoadNetworkRes[n].Road[r][l].BufferRoad[v].position;
											speedNew=0;
											accelerationNew=0;
										}
									}
									else//ǰ����δ�������������ֳ�ʼ״̬����
									{
										positionNew=RoadNetworkRes[n].Road[r][l].BufferRoad[v].position;
										speedNew=0;
										accelerationNew=0;
									}
								}
								else//�Ǹշ�������
								{
									//���¼��ٶ�
									if(RoadNetworkRes[n].Road[r][l].BufferRoad[v].speed==0 && RoadNetworkRes[n].Road[r][l].BufferRoad[v-1].position-RoadNetworkRes[n].Road[r][l].BufferRoad[v].position>5)
										accelerationNew=RoadNetworkRes[n].Road[r][l].BufferRoad[v].acceleration+1;
									else
										accelerationNew=RoadNetworkRes[n].Road[r][l].BufferRoad[v].speed*(RoadNetworkRes[n].Road[r][l].BufferRoad[v-1].speed-RoadNetworkRes[n].Road[r][l].BufferRoad[v].speed)/(RoadNetworkRes[n].Road[r][l].BufferRoad[v-1].position-RoadNetworkRes[n].Road[r][l].BufferRoad[v].position);

									//�����ٶ�
									speedNew=RoadNetworkRes[n].Road[r][l].BufferRoad[v].speed+RoadNetworkRes[n].Road[r][l].BufferRoad[v].acceleration;
									if (speedNew<0)
									{
										speedNew=0;
										accelerationNew=0;
									}

									//����λ��
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
								//������ʻ
							{
								//���¼��ٶ�
								if(RoadNetworkRes[n].Road[r][l].BufferRoad[v].speed<RoadNetworkRes[n].Road[r][l].BufferRoad[v].ExpSpeed)
									accelerationNew=RoadNetworkRes[n].Road[r][l].BufferRoad[v].acceleration+1;
								else
								{
									speedNew=RoadNetworkRes[n].Road[r][l].BufferRoad[v].ExpSpeed;
									accelerationNew=0;
								}


								//�����ٶ�
								speedNew=RoadNetworkRes[n].Road[r][l].BufferRoad[v].speed+RoadNetworkRes[n].Road[r][l].BufferRoad[v].acceleration;
								if (speedNew<0)
								{
									speedNew=0;
									accelerationNew=0;
								}

								//����λ��
								positionNew=RoadNetworkRes[n].Road[r][l].BufferRoad[v].position+RoadNetworkRes[n].Road[r][l].BufferRoad[v].speed+0.5*RoadNetworkRes[n].Road[r][l].BufferRoad[v].acceleration;
								if (positionNew<RoadNetworkRes[n].Road[r][l].BufferRoad[v].position || positionNew>=RoadNetworkRes[n].Road[r][l].BufferRoad[v-1].position+RoadNetworkRes[n].Road[r][l].BufferRoad[v-1].speed+0.5*RoadNetworkRes[n].Road[r][l].BufferRoad[v-1].acceleration*RoadNetworkRes[n].Road[r][l].BufferRoad[v-1].acceleration)
								{
									positionNew=RoadNetworkRes[n].Road[r][l].BufferRoad[v].position;
									speedNew=0;
									accelerationNew=0;
								}
							}

						}





						//������ͷ��״̬���·���2
						if (v==0 && RoadNetworkRes[n].Road[r][l].BufferRoad[RoadSIZE-1].VehicleFlag==true)
						{
							if (RoadNetworkRes[n].Road[r][l].BufferRoad[RoadSIZE-1].position<RoadNetworkRes[n].Road[r][l].BufferRoad[v].position)
								//��������
							{
								positionNew=RoadNetworkRes[n].Road[r][l].BufferRoad[v].position;
								speedNew=0;
								accelerationNew=0;
							}
							if (RoadNetworkRes[n].Road[r][l].BufferRoad[RoadSIZE-1].position-RoadNetworkRes[n].Road[r][l].BufferRoad[v].position<=150 && RoadNetworkRes[n].Road[r][l].BufferRoad[RoadSIZE-1].position-RoadNetworkRes[n].Road[r][l].BufferRoad[v].position>0)
								//����״̬
							{
								if (RoadNetworkRes[n].Road[r][l].BufferRoad[v].position==0)//�շ����ĳ���
								{
									if (RoadNetworkRes[n].Road[r][l].BufferRoad[RoadSIZE-1].position>0)//ǰ���Ѵ�������״̬�����η�����Ҫ�����ʼ�ٶȹ���
									{
										while(RoadNetworkRes[n].Road[r][l].BufferRoad[v].speed*1>RoadNetworkRes[n].Road[r][l].BufferRoad[RoadSIZE-1].position)
											//���������ٶ�
										{
											RoadNetworkRes[n].Road[r][l].BufferRoad[v].speed=RoadNetworkRes[n].Road[r][l].BufferRoad[v].speed/2;
										}
										//���¼��ٶ�
										if(RoadNetworkRes[n].Road[r][l].BufferRoad[v].speed==0 && RoadNetworkRes[n].Road[r][l].BufferRoad[RoadSIZE-1].position-RoadNetworkRes[n].Road[r][l].BufferRoad[v].position>5)
											accelerationNew=RoadNetworkRes[n].Road[r][l].BufferRoad[v].acceleration+1;
										else
											accelerationNew=RoadNetworkRes[n].Road[r][l].BufferRoad[v].speed*(RoadNetworkRes[n].Road[r][l].BufferRoad[RoadSIZE-1].speed-RoadNetworkRes[n].Road[r][l].BufferRoad[v].speed)/(RoadNetworkRes[n].Road[r][l].BufferRoad[RoadSIZE-1].position-RoadNetworkRes[n].Road[r][l].BufferRoad[v].position);

										//�����ٶ�
										speedNew=RoadNetworkRes[n].Road[r][l].BufferRoad[v].speed+RoadNetworkRes[n].Road[r][l].BufferRoad[v].acceleration;
										if (speedNew<0)
										{
											speedNew=0;
											accelerationNew=0;
										}

										//����λ��
										positionNew=RoadNetworkRes[n].Road[r][l].BufferRoad[v].position+RoadNetworkRes[n].Road[r][l].BufferRoad[v].speed+0.5*RoadNetworkRes[n].Road[r][l].BufferRoad[v].acceleration;
										if (positionNew<RoadNetworkRes[n].Road[r][l].BufferRoad[v].position || positionNew>=RoadNetworkRes[n].Road[r][l].BufferRoad[RoadSIZE-1].position+RoadNetworkRes[n].Road[r][l].BufferRoad[RoadSIZE-1].speed+0.5*RoadNetworkRes[n].Road[r][l].BufferRoad[RoadSIZE-1].acceleration*RoadNetworkRes[n].Road[r][l].BufferRoad[RoadSIZE-1].acceleration)
										{
											positionNew=RoadNetworkRes[n].Road[r][l].BufferRoad[v].position;
											speedNew=0;
											accelerationNew=0;
										}
									}
									else//ǰ����δ�������������ֳ�ʼ״̬����
									{
										positionNew=RoadNetworkRes[n].Road[r][l].BufferRoad[v].position;
										speedNew=0;
										accelerationNew=0;
									}
								}
								else//�Ǹշ�������
								{
									//���¼��ٶ�
									if(RoadNetworkRes[n].Road[r][l].BufferRoad[v].speed==0 && RoadNetworkRes[n].Road[r][l].BufferRoad[RoadSIZE-1].position-RoadNetworkRes[n].Road[r][l].BufferRoad[v].position>5)
										accelerationNew=RoadNetworkRes[n].Road[r][l].BufferRoad[v].acceleration+1;
									else
										accelerationNew=RoadNetworkRes[n].Road[r][l].BufferRoad[v].speed*(RoadNetworkRes[n].Road[r][l].BufferRoad[RoadSIZE-1].speed-RoadNetworkRes[n].Road[r][l].BufferRoad[v].speed)/(RoadNetworkRes[n].Road[r][l].BufferRoad[RoadSIZE-1].position-RoadNetworkRes[n].Road[r][l].BufferRoad[v].position);

									//�����ٶ�
									speedNew=RoadNetworkRes[n].Road[r][l].BufferRoad[v].speed+RoadNetworkRes[n].Road[r][l].BufferRoad[v].acceleration;
									if (speedNew<0)
									{
										speedNew=0;
										accelerationNew=0;
									}

									//����λ��
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
								//������ʻ
							{
								//���¼��ٶ�
								if(RoadNetworkRes[n].Road[r][l].BufferRoad[v].speed<RoadNetworkRes[n].Road[r][l].BufferRoad[v].ExpSpeed)
									accelerationNew=RoadNetworkRes[n].Road[r][l].BufferRoad[v].acceleration+1;
								else
								{
									speedNew=speedNew;
									accelerationNew=0;
								}


								//�����ٶ�
								speedNew=RoadNetworkRes[n].Road[r][l].BufferRoad[v].speed+RoadNetworkRes[n].Road[r][l].BufferRoad[v].acceleration;
								if (speedNew<0)
								{
									speedNew=0;
									accelerationNew=0;
								}

								//����λ��
								positionNew=RoadNetworkRes[n].Road[r][l].BufferRoad[v].position+RoadNetworkRes[n].Road[r][l].BufferRoad[v].speed+0.5*RoadNetworkRes[n].Road[r][l].BufferRoad[v].acceleration;
								if (positionNew<RoadNetworkRes[n].Road[r][l].BufferRoad[v].position || positionNew>=RoadNetworkRes[n].Road[r][l].BufferRoad[RoadSIZE-1].position+RoadNetworkRes[n].Road[r][l].BufferRoad[RoadSIZE-1].speed+0.5*RoadNetworkRes[n].Road[r][l].BufferRoad[RoadSIZE-1].acceleration*RoadNetworkRes[n].Road[r][l].BufferRoad[RoadSIZE-1].acceleration)
								{
									positionNew=RoadNetworkRes[n].Road[r][l].BufferRoad[v].position;
									speedNew=0;
									accelerationNew=0;
								}
							}
						}




						//����ͷ��״̬���·���
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
										accelerationNew=0;//�ٶȴﵽ�����ٶȺ�ת��������ʻ
									}


									speedNew=RoadNetworkRes[n].Road[r][l].BufferRoad[v].speed+RoadNetworkRes[n].Road[r][l].BufferRoad[v].acceleration*1;
									if(speedNew<0)
									{
										speedNew=0;
									}
									positionNew=RoadNetworkRes[n].Road[r][l].BufferRoad[v].position+RoadNetworkRes[n].Road[r][l].BufferRoad[v].speed*1+0.5*RoadNetworkRes[n].Road[r][l].BufferRoad[v].acceleration;//ʱ����Ϊ1��	
									if(positionNew<RoadNetworkRes[n].Road[r][l].BufferRoad[v-1].position)
									{
										positionNew=RoadNetworkRes[n].Road[r][l].BufferRoad[v].position;
									}
								}
								else//�������Ϊ���
								{
									if(VehicleLength*RoadSIZE-RoadNetworkRes[n].Road[r][l].BufferRoad[v].position>150)//�����·�ܳ������Ϊ0�����·�ܳ�Ϊ��ͨ������λ��
									{
										if(RoadNetworkRes[n].Road[r][l].BufferRoad[v].speed<RoadNetworkRes[n].Road[r][l].BufferRoad[v].ExpSpeed)
											accelerationNew=RoadNetworkRes[n].Road[r][l].BufferRoad[v].acceleration+1;
										else

										{
											speedNew=RoadNetworkRes[n].Road[r][l].BufferRoad[v].ExpSpeed;
											accelerationNew=0;//�ٶȴﵽ�����ٶȺ�ת��������ʻ
										}


										speedNew=RoadNetworkRes[n].Road[r][l].BufferRoad[v].speed+RoadNetworkRes[n].Road[r][l].BufferRoad[v].acceleration*1;
										if(speedNew<0)
										{
											speedNew=0;
										}
										positionNew=RoadNetworkRes[n].Road[r][l].BufferRoad[v].position+RoadNetworkRes[n].Road[r][l].BufferRoad[v].speed*1+0.5*RoadNetworkRes[n].Road[r][l].BufferRoad[v].acceleration;//ʱ����Ϊ1��	
										if(positionNew<RoadNetworkRes[n].Road[r][l].BufferRoad[v-1].position)
										{
											positionNew=RoadNetworkRes[n].Road[r][l].BufferRoad[v].position;
										}
									}
									else
									{
										//����GMģ�͸��¼��ٶ�
										accelerationNew=1*RoadNetworkRes[n].Road[r][l].BufferRoad[v].speed*(0-RoadNetworkRes[n].Road[r][l].BufferRoad[v].speed)/(VehicleLength*RoadSIZE-RoadNetworkRes[n].Road[r][l].BufferRoad[v].position);//1Ϊ��ӳǿ��ϵ��					

										//�����ٶ�				
										speedNew=RoadNetworkRes[n].Road[r][l].BufferRoad[v].speed+RoadNetworkRes[n].Road[r][l].BufferRoad[v].acceleration*1;
										if(speedNew<0)
										{
											speedNew=0;
										}

										//����λ��
										positionNew=RoadNetworkRes[n].Road[r][l].BufferRoad[v].position+RoadNetworkRes[n].Road[r][l].BufferRoad[v].speed+0.5*RoadNetworkRes[n].Road[r][l].BufferRoad[v].acceleration;//ʱ����Ϊ1��
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
										accelerationNew=0;//�ٶȴﵽ�����ٶȺ�ת��������ʻ
									}


									speedNew=RoadNetworkRes[n].Road[r][l].BufferRoad[v].speed+RoadNetworkRes[n].Road[r][l].BufferRoad[v].acceleration*1;
									if(speedNew<0)
									{
										speedNew=0;
									}
									positionNew=RoadNetworkRes[n].Road[r][l].BufferRoad[v].position+RoadNetworkRes[n].Road[r][l].BufferRoad[v].speed*1+0.5*RoadNetworkRes[n].Road[r][l].BufferRoad[v].acceleration;//ʱ����Ϊ1��	
									if(positionNew<RoadNetworkRes[n].Road[r][l].BufferRoad[v-1].position)
									{
										positionNew=RoadNetworkRes[n].Road[r][l].BufferRoad[v].position;
									}
								}
								else
								{
									if(VehicleLength*RoadSIZE-RoadNetworkRes[n].Road[r][l].BufferRoad[v].position>150)//�����·�ܳ������Ϊ0�����·�ܳ�Ϊ��ͨ������λ��
									{
										if(RoadNetworkRes[n].Road[r][l].BufferRoad[v].speed<RoadNetworkRes[n].Road[r][l].BufferRoad[v].ExpSpeed)
											accelerationNew=RoadNetworkRes[n].Road[r][l].BufferRoad[v].acceleration+1;
										else

										{
											speedNew=RoadNetworkRes[n].Road[r][l].BufferRoad[v].ExpSpeed;
											accelerationNew=0;//�ٶȴﵽ�����ٶȺ�ת��������ʻ
										}


										speedNew=RoadNetworkRes[n].Road[r][l].BufferRoad[v].speed+RoadNetworkRes[n].Road[r][l].BufferRoad[v].acceleration*1;
										if(speedNew<0)
										{
											speedNew=0;
										}
										positionNew=RoadNetworkRes[n].Road[r][l].BufferRoad[v].position+RoadNetworkRes[n].Road[r][l].BufferRoad[v].speed*1+0.5*RoadNetworkRes[n].Road[r][l].BufferRoad[v].acceleration;//ʱ����Ϊ1��	
										if(positionNew<RoadNetworkRes[n].Road[r][l].BufferRoad[v-1].position)
										{
											positionNew=RoadNetworkRes[n].Road[r][l].BufferRoad[v].position;
										}
									}
									else
									{
										//����GMģ�͸��¼��ٶ�
										accelerationNew=1*RoadNetworkRes[n].Road[r][l].BufferRoad[v].speed*(0-RoadNetworkRes[n].Road[r][l].BufferRoad[v].speed)/(VehicleLength*RoadSIZE-RoadNetworkRes[n].Road[r][l].BufferRoad[v].position);//1Ϊ��ӳǿ��ϵ��					

										//�����ٶ�				
										speedNew=RoadNetworkRes[n].Road[r][l].BufferRoad[v].speed+RoadNetworkRes[n].Road[r][l].BufferRoad[v].acceleration*1;
										if(speedNew<0)
										{
											speedNew=0;
										}

										//����λ��
										positionNew=RoadNetworkRes[n].Road[r][l].BufferRoad[v].position+RoadNetworkRes[n].Road[r][l].BufferRoad[v].speed+0.5*RoadNetworkRes[n].Road[r][l].BufferRoad[v].acceleration;//ʱ����Ϊ1��
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
										accelerationNew=0;//�ٶȴﵽ�����ٶȺ�ת��������ʻ
									}


									speedNew=RoadNetworkRes[n].Road[r][l].BufferRoad[v].speed+RoadNetworkRes[n].Road[r][l].BufferRoad[v].acceleration*1;
									if(speedNew<0)
									{
										speedNew=0;
									}
									positionNew=RoadNetworkRes[n].Road[r][l].BufferRoad[v].position+RoadNetworkRes[n].Road[r][l].BufferRoad[v].speed*1+0.5*RoadNetworkRes[n].Road[r][l].BufferRoad[v].acceleration;//ʱ����Ϊ1��	
									if(positionNew<RoadNetworkRes[n].Road[r][l].BufferRoad[v-1].position)
									{
										positionNew=RoadNetworkRes[n].Road[r][l].BufferRoad[v].position;
									}
								}
								else
								{
									if(VehicleLength*RoadSIZE-RoadNetworkRes[n].Road[r][l].BufferRoad[v].position>150)//�����·�ܳ������Ϊ0�����·�ܳ�Ϊ��ͨ������λ��
									{
										if(RoadNetworkRes[n].Road[r][l].BufferRoad[v].speed<RoadNetworkRes[n].Road[r][l].BufferRoad[v].ExpSpeed)
											accelerationNew=RoadNetworkRes[n].Road[r][l].BufferRoad[v].acceleration+1;
										else

										{
											speedNew=RoadNetworkRes[n].Road[r][l].BufferRoad[v].ExpSpeed;
											accelerationNew=0;//�ٶȴﵽ�����ٶȺ�ת��������ʻ

										}

										speedNew=RoadNetworkRes[n].Road[r][l].BufferRoad[v].speed+RoadNetworkRes[n].Road[r][l].BufferRoad[v].acceleration*1;
										if(speedNew<0)
										{
											speedNew=0;
										}
										positionNew=RoadNetworkRes[n].Road[r][l].BufferRoad[v].position+RoadNetworkRes[n].Road[r][l].BufferRoad[v].speed*1+0.5*RoadNetworkRes[n].Road[r][l].BufferRoad[v].acceleration;//ʱ����Ϊ1��	
										if(positionNew<RoadNetworkRes[n].Road[r][l].BufferRoad[v-1].position)
										{
											positionNew=RoadNetworkRes[n].Road[r][l].BufferRoad[v].position;
										}
									}
									else
									{
										//����GMģ�͸��¼��ٶ�
										accelerationNew=1*RoadNetworkRes[n].Road[r][l].BufferRoad[v].speed*(0-RoadNetworkRes[n].Road[r][l].BufferRoad[v].speed)/(VehicleLength*RoadSIZE-RoadNetworkRes[n].Road[r][l].BufferRoad[v].position);//1Ϊ��ӳǿ��ϵ��					

										//�����ٶ�				
										speedNew=RoadNetworkRes[n].Road[r][l].BufferRoad[v].speed+RoadNetworkRes[n].Road[r][l].BufferRoad[v].acceleration*1;
										if(speedNew<0)
										{
											speedNew=0;
										}

										//����λ��
										positionNew=RoadNetworkRes[n].Road[r][l].BufferRoad[v].position+RoadNetworkRes[n].Road[r][l].BufferRoad[v].speed+0.5*RoadNetworkRes[n].Road[r][l].BufferRoad[v].acceleration;//ʱ����Ϊ1��
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
										accelerationNew=0;//�ٶȴﵽ�����ٶȺ�ת��������ʻ

									speedNew=RoadNetworkRes[n].Road[r][l].BufferRoad[v].speed+RoadNetworkRes[n].Road[r][l].BufferRoad[v].acceleration*1;
									if(speedNew<0)
									{
										speedNew=0;
									}
									positionNew=RoadNetworkRes[n].Road[r][l].BufferRoad[v].position+RoadNetworkRes[n].Road[r][l].BufferRoad[v].speed*1+0.5*RoadNetworkRes[n].Road[r][l].BufferRoad[v].acceleration;//ʱ����Ϊ1��	
									if(positionNew<RoadNetworkRes[n].Road[r][l].BufferRoad[v-1].position)
									{
										positionNew=RoadNetworkRes[n].Road[r][l].BufferRoad[v].position;
									}
								}
								else
								{
									if(VehicleLength*RoadSIZE-RoadNetworkRes[n].Road[r][l].BufferRoad[v].position>150)//�����·�ܳ������Ϊ0�����·�ܳ�Ϊ��ͨ������λ��
									{
										if(RoadNetworkRes[n].Road[r][l].BufferRoad[v].speed<RoadNetworkRes[n].Road[r][l].BufferRoad[v].ExpSpeed)
											accelerationNew=RoadNetworkRes[n].Road[r][l].BufferRoad[v].acceleration+1;
										else
											accelerationNew=0;//�ٶȴﵽ�����ٶȺ�ת��������ʻ

										speedNew=RoadNetworkRes[n].Road[r][l].BufferRoad[v].speed+RoadNetworkRes[n].Road[r][l].BufferRoad[v].acceleration*1;
										if(speedNew<0)
										{
											speedNew=0;
										}
										positionNew=RoadNetworkRes[n].Road[r][l].BufferRoad[v].position+RoadNetworkRes[n].Road[r][l].BufferRoad[v].speed*1+0.5*RoadNetworkRes[n].Road[r][l].BufferRoad[v].acceleration;//ʱ����Ϊ1��	
										if(positionNew<RoadNetworkRes[n].Road[r][l].BufferRoad[v-1].position)
										{
											positionNew=RoadNetworkRes[n].Road[r][l].BufferRoad[v].position;
										}
									}
									else
									{
										//����GMģ�͸��¼��ٶ�
										accelerationNew=1*RoadNetworkRes[n].Road[r][l].BufferRoad[v].speed*(0-RoadNetworkRes[n].Road[r][l].BufferRoad[v].speed)/(VehicleLength*RoadSIZE-RoadNetworkRes[n].Road[r][l].BufferRoad[v].position);//1Ϊ��ӳǿ��ϵ��					

										//�����ٶ�				
										speedNew=RoadNetworkRes[n].Road[r][l].BufferRoad[v].speed+RoadNetworkRes[n].Road[r][l].BufferRoad[v].acceleration*1;
										if(speedNew<0)
										{
											speedNew=0;
										}

										//����λ��
										positionNew=RoadNetworkRes[n].Road[r][l].BufferRoad[v].position+RoadNetworkRes[n].Road[r][l].BufferRoad[v].speed+0.5*RoadNetworkRes[n].Road[r][l].BufferRoad[v].acceleration;//ʱ����Ϊ1��
										if(positionNew<RoadNetworkRes[n].Road[r][l].BufferRoad[v].position)
										{
											positionNew=RoadNetworkRes[n].Road[r][l].BufferRoad[v].position;
										}	

									}

								}
							}
						}

						//�����ݸ�ֵ

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
	if (i==0)//��
	{
		if (j==0)//��ת
		{
			destinationRoadNum=3;
		}
		if (j==1)//ֱ��
		{
			destinationRoadNum=2;
		}
		if (j==2)//��ת
		{
			destinationRoadNum=1;
		}
	}
	if (i==1)//��
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
	if (i==2)//��
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
	if (i==3)//��
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
//ÿ������ʱ�̵ķ������ݺ���
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

				destinationRoad=GetODList(r,l);//���ڵ��ڵ�Ŀ��·��
				if (RoadNetworkRes[n].NodeLinkTo[destinationRoad]==-1)
				//ʻ��·�� ɾ������
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
				//�ӱ���·ʻ��������·
				{
					int front=RoadNetworkRes[n].Road[r][l].RoadFront;
					if (RoadNetworkRes[n].Road[r][l].BufferRoad[front].position>RoadSIZE*VehicleLength+1)
					{
						NodeNumTemp=RoadNetworkRes[n].NodeLinkTo[destinationRoad];//Ŀ��ڵ��
						RoadNumTemp=RoadNetworkRes[n].RoadLinkTo[destinationRoad];//Ŀ��ڵ��Ŀ���·
						
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

						//ԭ��������ɾ��
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

						//�����µ�Ŀ���·(rear+1)%RoadSIZE!=front
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
							RoadNetworkRes[NodeNumTemp].Road[RoadNumTemp][LaneNumTemp].RoadRear=(RoadNetworkRes[NodeNumTemp].Road[RoadNumTemp][LaneNumTemp].RoadRear+1)%RoadSIZE;//ѭ�����еĲ���
							RoadNetworkRes[NodeNumTemp].Road[RoadNumTemp][LaneNumTemp].RoadCount++;
						}
						
					}
					
				}

			}
		}
	}
	
}

void DateProcessEnd()
//���յ����ݴ������������ݣ�����·�ι�����
{
	for (int n=0;n<NodeNumTol;n++)
	{
		for (int r=0;r<NumofRoads;r++)
		{
			/*//����t
			RoadNetworkRes[n].RoadTraveTime[r]=(float)(RoadNetworkRes[n].RoadTraveTime[r]/(60*RoadNetworkRes[n].VehicleInNumInAll[r]));
			//����Qmax
			int temp=-1;
			for (int l=0;l<NumofLanes;l++)
			{
				if (RoadNetworkRes[n].RoadVehiclePassNum[r][l]>temp)
				{
					RoadNetworkRes[n].VehicleInNumInMax[r]=RoadNetworkRes[n].RoadVehiclePassNum[r][l];
					temp=RoadNetworkRes[n].RoadVehiclePassNum[r][l];
				}
			}
			//����Qtotal
			RoadNetworkRes[n].VehicleInNumInAll[r]=(int)(RoadNetworkRes[n].VehicleInNumInAll[r]/NumofLanes);
			float ratio=(float)RoadNetworkRes[n].VehicleInNumInMax[r]/RoadNetworkRes[n].VehicleInNumInAll[r];

			/////////�����·�ι�����
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

	//��дlaplace����
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


	//д��ͼ����
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

	//д�����������ļ�
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
		//����
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

		//���ַ�����Ӧ��������Ŀ����ֵ
		PeopleCurrent[i].N=0;
		PeopleNew[i].N=0;
		//���ַ�����Ӧ��������Ӧ�����鸶��ֵ
		for(int sa=0;sa<NodeNumTol;sa++)	
		{
			for (int sa2=0;sa2<NodeNumTol;sa2++)
			{
				PeopleCurrent[i].NRelation[sa][sa2]=-1;
				PeopleNew[i].NRelation[sa][sa2]=-1;
			}			
		}
		
		//���ַ�����Ӧ�ȸ���ֵ
		PeopleCurrent[i].fitness=0;
		PeopleNew[i].fitness=0;


		//����Ⱦɫ�������
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

	//�����뻮�ַ���������صı�����ʼ��
	for (int i=0;i<PeopleNum;i++)
	{
		//���ַ�����Ӧ��������Ŀ����ֵ
		PeopleCurrent[i].N=0;
		//���ַ�����Ӧ��������Ӧ�����鸶��ֵ
		for(int sa=0;sa<NodeNumTol;sa++)	
		{
			for (int sa2=0;sa2<NodeNumTol;sa2++)
			{
				PeopleCurrent[i].NRelation[sa][sa2]=-1;
			}

		}
		//���ַ�����Ӧ�ȸ���ֵ
		PeopleCurrent[i].fitness=0;
	}


	int TempIdx=0;
	//����
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

	//��PeopleCurrent[i].LapMatrixBinary[ii][jj]Ϊ�����б����������Լ���������� �Ӷ��ó�����������ֵ
	bool CheckFlag=true;
	int roottemp=-1;
	int treeNumber;
	int root;

	int TreeIdx;
	//�洢���ڵ��Ҷ�ڵ����ݵ�һά������������
	int LeafIdx;
	//ȷ����ǰ���ĸ��ڵ�			
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
		
		while (root!=-1)//��������������ͼ���
		{
			//���������Ļ�����Ŀ
			//PeopleCurrent[i].N=PeopleCurrent[i].N+1;

			//һ��ѭ������һ���� �洢�ṹΪһά����tree

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


			//�洢�����нڵ����ݵ�һά������������
			 TreeIdx=1;
			//�洢���ڵ��Ҷ�ڵ����ݵ�һά������������
			 LeafIdx=0;
			//ȷ����ǰ���ĸ��ڵ�			
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
			
			// Ѱ�ҵ����������нڵ�
			while (rootIdx<TreeIdx)
			{
				// ���ڵ��Ӧ��Ҷ�ڵ���Ÿ�ֵ
				for (int k=0;k<NodeNumTol;k++)
				{
					if (PeopleCurrent[i].LapMatrixBinary[roottemp][k]==1 && NodeFlag[k]==true)
					{	
						tree[rootIdx].Leaf[LeafIdx]=k;
						LeafIdx++;
					}
				}
				//�ҵ���ǰroot���е�leaf֮��root��־״̬����
				NodeFlag[roottemp]=false;
				
				//��root��leaf�ڵ����tree����
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

				//�ض�λroot
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


		//���㲻ͬ���ַ�������
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

			//��temp���и�ֵ������
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
			//����
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

			//�������������Ϲ�����
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
	//����
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
	//����
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
		//����Ⱦɫ�������
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

	//�����뻮�ַ���������صı�����ʼ��
	for (int i=0;i<PeopleNum;i++)
	{
		//���ַ�����Ӧ��������Ŀ����ֵ
		PeopleNew[i].N=0;
		//���ַ�����Ӧ��������Ӧ�����鸳��ֵ
		for(int sa=0;sa<NodeNumTol;sa++)	
		{
			for (int sa2=0;sa2<NodeNumTol;sa2++)
			{
				PeopleNew[i].NRelation[sa][sa2]=-1;
			}
		}
		//���ַ�����Ӧ�ȸ���ֵ
		PeopleNew[i].fitness=0;
	}


	int TempIdx=0;
	//����
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

	//��PeopleCurrent[i].LapMatrixBinary[ii][jj]Ϊ�����б����������Լ���������� �Ӷ��ó�����������ֵ
	for (int NodeFlagIdx=0;NodeFlagIdx<NodeNumTol;NodeFlagIdx++)
	{
		NodeFlag[NodeFlagIdx]=true;
	}

	bool CheckFlag=true;
	int roottemp=-1;
	int root;
	int treeNumber;

	//�洢�����нڵ����ݵ�һά������������
	int TreeIdx;
	//�洢���ڵ��Ҷ�ڵ����ݵ�һά������������
	int LeafIdx;
	//ȷ����ǰ���ĸ��ڵ�			
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
		
		while (root!=-1)//��������������ͼ���
		{
			//���������Ļ�����Ŀ
			//PeopleCurrent[i].N=PeopleCurrent[i].N+1;

			//һ��ѭ������һ���� �洢�ṹΪһά����tree
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


			//�洢�����нڵ����ݵ�һά������������
			TreeIdx=1;
			//�洢���ڵ��Ҷ�ڵ����ݵ�һά������������
			LeafIdx=0;
			//ȷ����ǰ���ĸ��ڵ�			
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

			// Ѱ�ҵ����������нڵ�
			while (rootIdx<TreeIdx)
			{
				// ���ڵ��Ӧ��Ҷ�ڵ���Ÿ�ֵ
				for (int k=0;k<NodeNumTol;k++)
				{
					if (PeopleNew[i].LapMatrixBinary[roottemp][k]==1 && NodeFlag[k]==true)
					{	
						tree[rootIdx].Leaf[LeafIdx]=k;
						LeafIdx++;
					}
				}
				//�ҵ���ǰroot���е�leaf֮��root��־״̬����
				NodeFlag[roottemp]=false;

				//��root��leaf�ڵ����tree����
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

				//�ض�λroot
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


		//���㲻ͬ���ַ�������
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

			//��temp���и�ֵ������
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
			//����
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

			//�������������Ϲ�����
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
//һ��ͨ���������·���и�·�ι�����
	

	int ComputeMode;
	//0��CPU����ģʽ  ������GPU����ģʽ
	ComputeMode=1;

	//��ʼ��·������
	RoadNetworkInitial();

	//��ͨ�����ݳ�ʼ��
	TrafficLightsDataInitial();
	//·�����ݳ�ʼ��
	RoadNetworkDataInitial();


	//���ɽ�ͨ���Ʒ���
	//TrafficControlProGen();
	TrafficControlProGen2();

	SimulationClock=0;
	SimulationDuration=10800;

	//FILE * recordTime;
	//recordTime=fopen("Count.txt","w+");

	while (SimulationClock<SimulationDuration)
	{
		//1��ˢ�½�ͨ��״̬
		TrafficLightsRefresh();	
		//2�����ճ����ĳ���ʱ�佫�������뵽·���� 
		VehicleGeneration();
		//3��������溯��--���ݡ��������͡�������ԭ�����·���г���״̬
		SimulationProcess();
		//4�����ݺ���--��������λ�õĸı�
		DataPorcess();

		//fprintf(recordTime,"%d \n",RoadNetworkRes[7].Road[2][1].RoadCount);

		SimulationClock++;
		//����Ļ����ʾ��ǰ�ķ���ʱ��
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

	//���յ����ݴ������������ݣ�����·�ι�����
	DateProcessEnd();
	
///////////////////////////////////////////////////////////
//������ʼӦ��GA���п�����������
	int TempIdx=0;
	for (int i=0;i<NodeNumTol;i++)
	{
		for (int j=0;j<NodeNumTol;j++)
		{
			if (LapMatirx[i][j]!=-1  && j>i)//�������Ǿ�����ȡֵ
			{
				LapVector[TempIdx]=LapMatirx[i][j];
				TempIdx++;
			}
		}
	}

	printf("\n");
	printf("Begin to divide the control area into subareas...\n");
	int iteration=0;

	//���ɳ�ʼ��Ⱥ P
	PeopleInitial();
	//������Ⱥ P ��Ӧ�� 
	EvaluatePeopleCurrent();

	FILE * DivideIteration;
	DivideIteration=fopen("DivideIteration.txt","w+");

	while(iteration<MaxGen)
	{
		//����������� W
		CrossoverAndMutationOperation();
		//������Ⱥ W ��Ӧ��
		EvaluatePeopleNew();
		//ѡ����� ���� P �еĻ��ַ�������
		SelectionOpration();

		iteration++;

		HANDLE hOut=GetStdHandle(STD_OUTPUT_HANDLE);
		COORD pos0;
		pos0.X=0;
		pos0.Y=2;
		SetConsoleCursorPosition(hOut,pos0);
		fprintf(stdout,"Division: CurrentIteration is %d",iteration);

		//д�ļ�
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

	
	//��ӡ��ѻ��ַ���
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
