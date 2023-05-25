function [cir, trans_delay, multipath_delay] = generate_cir(filename,fs)
% cir--������ŵ��弤��Ӧ��trans_delay--����ʱ�ӣ�multipath_delay--�ྶʱ��
% filename = "Testsd_00050.arr"--����ʱ��-���ȵ��ļ�
% fs--����Ƶ��

[ amp, delay, ~, ~, ~, ~, ~, ~ ] = read_arrivals_asc(  filename ); % ��ȡʱ�Ӻͷ���
% ��Ϊamp��delay���ж����0����Ҫ����ȥ��
% [m,n] = size(amp); % amp��delay����ͬ��С�ľ���,m = 1, n = 200
delay = delay(delay ~= 0);
amp = abs(amp(amp ~= 0));
trans_delay = min(delay);
multipath_delay = max(delay) - min(delay);
sampling_delay = ceil(delay*fs); % ����ȡ��

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