シグモイド関数の定義
$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

分母を y とおく
$$y = 1 + e^{-x}$$
シグモイド関数は次のように書き直される

$$\sigma(x) = \frac{1}{y}$$

逆数の微分法則の使用

$$\frac{d}{dx}\left(\frac{1}{y}\right) = -\frac{1}{y^2} \cdot \frac{dy}{dx}$$

$$\frac{d}{dx}\left(\frac{1}{y}\right) = \frac{d}{dy}\left(\frac{1}{y}\right) \cdot \frac{dy}{dx}$$

$y = 1 + e^{-x}$ の微分

$$\frac{dy}{dx} = \frac{d}{dx}(1 + e^{-x}) = -e^{-x}$$

微分の組み合わせ

$$\sigma'(x) = -\frac{1}{(1 + e^{-x})^2} \cdot (-e^{-x})$$

分母と分子を簡略化

$$\sigma'(x) = \frac{e^{-x}}{(1 + e^{-x})^2}$$

シグモイド関数を元に戻す

$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

$$1 - \sigma(x) = \frac{e^{-x}}{1 + e^{-x}}$$

最終的な微分
$$\sigma'(x) = \sigma(x) \cdot (1 - \sigma(x))$$
