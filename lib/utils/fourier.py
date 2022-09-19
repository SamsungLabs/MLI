"""
Fourier operations, consistent both with old and new pytorch fft interfaces.
We follow the conventions of the old interface dor simplicity
"""

from typing import NamedTuple, Optional, Union

from packaging import version
import torch


NEW_INTERFACE = version.parse(torch.__version__) > version.parse('1.6')


class FourierOutput(NamedTuple):
    real: torch.Tensor
    imag: Optional[torch.Tensor] = None

    def convert_to_complex_tensor(self):
        if self.imag is None:
            return self.real
        else:
            return torch.complex(self.real, self.imag)

    def convert_to_stacked_tensor(self):
        if self.imag is None:
            imag = torch.zeros_like(self.real)
            return torch.stack([self.real, imag], dim=-1)
        else:
            return torch.stack([self.real, self.imag], dim=-1)


def fft(input: Union[torch.Tensor, FourierOutput],
        signal_ndim: int = 1,
        normalized: bool = False,
        ) -> FourierOutput:
    assert signal_ndim in (1, 2, 3)
    if NEW_INTERFACE:
        if isinstance(input, FourierOutput):
            input = input.convert_to_complex_tensor()
        out = torch.fft.fftn(input,
                             dim=tuple(range(-signal_ndim, 0)),
                             norm='ortho' if normalized else 'backward',
                             )
        return FourierOutput(out.real, out.imag)
    else:
        if isinstance(input, FourierOutput):
            input = input.convert_to_stacked_tensor()
        assert input.shape[-1] == 2
        return FourierOutput(*torch.fft(input, signal_ndim=signal_ndim, normalized=normalized).unbind(-1))


def ifft(input: Union[torch.Tensor, FourierOutput],
         signal_ndim: int = 1,
         normalized: bool = False,
         ) -> FourierOutput:
    assert signal_ndim in (1, 2, 3)
    if NEW_INTERFACE:
        if isinstance(input, FourierOutput):
            input = input.convert_to_complex_tensor()
        out = torch.fft.ifftn(input,
                              dim=tuple(range(-signal_ndim, 0)),
                              normalized='ortho' if normalized else 'forward',
                              )
        return FourierOutput(out.real, out.imag)
    else:
        if isinstance(input, FourierOutput):
            input = input.convert_to_stacked_tensor()
        assert input.shape[-1] == 2
        return FourierOutput(*torch.ifft(input, signal_ndim=signal_ndim, normalized=normalized).unbind(-1))


def rfft(input: Union[torch.Tensor, FourierOutput],
         signal_ndim: int = 1,
         normalized: bool = False,
         ) -> FourierOutput:
    assert signal_ndim in (1, 2, 3)
    if NEW_INTERFACE:
        return fft(input, signal_ndim=signal_ndim, normalized=normalized)
    else:
        if isinstance(input, FourierOutput):
            input = input.convert_to_stacked_tensor()
        return FourierOutput(
            *torch.rfft(input, signal_ndim=signal_ndim, normalized=normalized, onesided=False).unbind(-1))


def irfft(input,
          signal_ndim: int = 1,
          normalized: bool = False,
          ) -> FourierOutput:
    assert signal_ndim in (1, 2, 3)
    if NEW_INTERFACE:
        if isinstance(input, FourierOutput):
            input = input.convert_to_complex_tensor()
        out = torch.fft.ifftn(input,
                              dim=tuple(range(-signal_ndim, 0)),
                              normalized='ortho' if normalized else 'forward',
                              )
        return FourierOutput(out.real)
    else:
        if isinstance(input, FourierOutput):
            input = input.convert_to_stacked_tensor()
        return FourierOutput(torch.irfft(input, signal_ndim=signal_ndim, normalized=normalized, onesided=False))
