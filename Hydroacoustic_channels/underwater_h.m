%����ˮ���ŵ��弤��Ӧ
filename1 = 'Testsd_00100.arr';
filename2 = 'Testrd_00100.arr';
fs = 1000; % ����Ƶ�ʡ���Ϣ����
[cir_sd, trans_delay_sd, multipath_delay1] = generate_cir(filename1,fs);
[cir_rd, trans_delay_rd, multipath_delay2] = generate_cir(filename2,fs);
%������Դ�ڵ㵽Ŀ�Ľڵ��ˮ���ŵ�
cir_sd=cir_sd*1e3;                                                                                                                                                             cir_sd=cir_sd(1:8);cir_sd=[0.8716 0 0 0 0.3175 0 0 -0.1932]+0*cir_sd;
%�������м̽ڵ㵽Ŀ�Ľڵ��ˮ���ŵ�
cir_rd=cir_rd*1e3;                                                                                                                                                             cir_rd=cir_rd(1:12);cir_rd=[0.5550,-0.5144,0.5782,0,0,-0.3051,0,0,0,0,0,0.52]+0*cir_rd;
save cir_sd.mat cir_sd
save cir_rd.mat cir_rd