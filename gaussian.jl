using Plots

# Gaussian distribution 함수 정의
function gaussian(A, x, μ, σ)
    return (A / (σ * sqrt(2 * π))) * exp(-0.5 * ((x - μ) / σ)^2)
end

# 파라미터 설정
A1 = 1       # 진폭
A2 = 0.8       # 진폭 
A3 = 0.8       # 진폭 
A4 = 0.6       # 진폭 
A5 = 0.6       # 진폭 

μ1 = 0       # 평균
μ2 = -1      # 평균
μ3 = 1       # 평균
μ4 = -2      # 평균
μ5 = 2       # 평균

σ1 = 0.4       # 표준편차
σ2 = 0.4       # 표준편차
σ3 = 0.4       # 표준편차
σ4 = 0.4       # 표준편차
σ5 = 0.4       # 표준편차

x = -5:0.1:5  # x 범위

# Gaussian 분포 계산
y1 = [gaussian(A1, xi, μ1, σ1) for xi in x]
y2 = [gaussian(A2, xi, μ2, σ2) for xi in x]
y3 = [gaussian(A3, xi, μ3, σ3) for xi in x]
y4 = [gaussian(A4, xi, μ4, σ4) for xi in x]
y5 = [gaussian(A5, xi, μ5, σ5) for xi in x]

y = y1 .+ y2 .+ y3 .+ y4 .+ y5 # Gaussian 분포 합성

# Plot 그리기
p = plot(x, y, 
        #  label="Gaussian Distribution", 
        #  xlabel="x", 
        #  ylabel="Density", 
        #  title="Gaussian Distribution", 
         color=:blue, 
         lw=2, 
         dpi=300,
         bg=:Transparent
         )

# Plot 저장
savefig(p, "gaussian_distribution.png")