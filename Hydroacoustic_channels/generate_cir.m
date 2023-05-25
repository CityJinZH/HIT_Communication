function [cir, trans_delay, multipath_delay] = generate_cir(filename,fs)
% cir--输出的信道冲激响应，trans_delay--传输时延，multipath_delay--多径时延
% filename = "Testsd_00050.arr"--存在时延-幅度的文件
% fs--采样频率

[ amp, delay, ~, ~, ~, ~, ~, ~ ] = read_arrivals_asc(  filename ); % 读取时延和幅度
% 因为amp和delay中有多余的0，需要将其去除
% [m,n] = size(amp); % amp和delay是相同大小的矩阵,m = 1, n = 200
delay = delay(delay ~= 0);
amp = abs(amp(amp ~= 0));
trans_delay = min(delay);
multipath_delay = max(delay) - min(delay);
sampling_delay = ceil(delay*fs); % 向上取整

for i = 2:length(amp)
    for j = 1:(i-1)
        if sampling_delay(j) == sampling_delay(i)
            sampling_delay(i) = 0;
            amp(i) = 0;
        end
    end
end

sampling_delay = sampling_delay(sampling_delay ~= 0);
amp = amp(amp ~= 0);
delay_tag = ceil(sampling_delay-ceil(trans_delay*fs)) + 1;

for i = 1:length(delay_tag)
    cir(delay_tag(i)) = amp(i);
end

end