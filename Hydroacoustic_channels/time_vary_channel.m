load cir_sd.mat 
load cir_rd.mat 
h1=zeros(3000,length(cir_sd));%%��ʼ�����м̽ڵ㵽Ŀ�Ľڵ���ŵ�1
h2=zeros(3000,length(cir_rd));%%��ʼ����Դ�ӵ��ǽڵ㵽Ŀ�Ľڵ���ŵ�2
h1(1,:)=cir_sd;
h2(1,:)=cir_rd;
w=0.1;%��������չ
a=2-cos(w./2)-sqrt((cos(w./2).^2).^2-4*cos(w./2)+3);%ʱ������
for j=1:2999
    for i=1:length(cir_sd)
        if h1(j,i)~=0
           h1(j+1,i)=h1(j,i)*a+sqrt(1-a^2)*normrnd(0,1);
        end
   end
   for ii=1:length(cir_rd)
        if h2(j,ii)~=0
           h2(j+1,ii)=h2(j,ii)*a+sqrt(1-a^2)*normrnd(0,1);
        end
   end
end
save h1.mat h1
save h2.mat h2