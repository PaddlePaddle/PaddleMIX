# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle


class Sequential(paddle.nn.Layer):
    def __init__(self):
        super(Sequential, self).__init__()
        self.conv2d0 = paddle.nn.Conv2D(
            stride=2, padding=3, out_channels=64, kernel_size=(7, 7), bias_attr=False, in_channels=3
        )
        self.batchnorm0 = paddle.nn.BatchNorm(is_test=True, num_channels=64, momentum=0.1)
        self.relu0 = paddle.nn.ReLU()
        self.pool2d0 = paddle.nn.MaxPool2D(kernel_size=[3, 3], stride=2, padding=1)
        self.conv2d1 = paddle.nn.Conv2D(out_channels=64, kernel_size=(1, 1), bias_attr=False, in_channels=64)
        self.batchnorm1 = paddle.nn.BatchNorm(is_test=True, num_channels=64, momentum=0.1)
        self.relu1 = paddle.nn.ReLU()
        self.conv2d2 = paddle.nn.Conv2D(
            padding=1, out_channels=64, kernel_size=(3, 3), bias_attr=False, in_channels=64
        )
        self.batchnorm2 = paddle.nn.BatchNorm(is_test=True, num_channels=64, momentum=0.1)
        self.relu2 = paddle.nn.ReLU()
        self.conv2d3 = paddle.nn.Conv2D(out_channels=256, kernel_size=(1, 1), bias_attr=False, in_channels=64)
        self.batchnorm3 = paddle.nn.BatchNorm(is_test=True, num_channels=256, momentum=0.1)
        self.conv2d4 = paddle.nn.Conv2D(out_channels=256, kernel_size=(1, 1), bias_attr=False, in_channels=64)
        self.batchnorm4 = paddle.nn.BatchNorm(is_test=True, num_channels=256, momentum=0.1)
        self.relu3 = paddle.nn.ReLU()
        self.conv2d5 = paddle.nn.Conv2D(out_channels=64, kernel_size=(1, 1), bias_attr=False, in_channels=256)
        self.batchnorm5 = paddle.nn.BatchNorm(is_test=True, num_channels=64, momentum=0.1)
        self.relu4 = paddle.nn.ReLU()
        self.conv2d6 = paddle.nn.Conv2D(
            padding=1, out_channels=64, kernel_size=(3, 3), bias_attr=False, in_channels=64
        )
        self.batchnorm6 = paddle.nn.BatchNorm(is_test=True, num_channels=64, momentum=0.1)
        self.relu5 = paddle.nn.ReLU()
        self.conv2d7 = paddle.nn.Conv2D(out_channels=256, kernel_size=(1, 1), bias_attr=False, in_channels=64)
        self.batchnorm7 = paddle.nn.BatchNorm(is_test=True, num_channels=256, momentum=0.1)
        self.relu6 = paddle.nn.ReLU()
        self.conv2d8 = paddle.nn.Conv2D(out_channels=64, kernel_size=(1, 1), bias_attr=False, in_channels=256)
        self.batchnorm8 = paddle.nn.BatchNorm(is_test=True, num_channels=64, momentum=0.1)
        self.relu7 = paddle.nn.ReLU()
        self.conv2d9 = paddle.nn.Conv2D(
            padding=1, out_channels=64, kernel_size=(3, 3), bias_attr=False, in_channels=64
        )
        self.batchnorm9 = paddle.nn.BatchNorm(is_test=True, num_channels=64, momentum=0.1)
        self.relu8 = paddle.nn.ReLU()
        self.conv2d10 = paddle.nn.Conv2D(out_channels=256, kernel_size=(1, 1), bias_attr=False, in_channels=64)
        self.batchnorm10 = paddle.nn.BatchNorm(is_test=True, num_channels=256, momentum=0.1)
        self.relu9 = paddle.nn.ReLU()
        self.conv2d11 = paddle.nn.Conv2D(out_channels=128, kernel_size=(1, 1), bias_attr=False, in_channels=256)
        self.batchnorm11 = paddle.nn.BatchNorm(is_test=True, num_channels=128, momentum=0.1)
        self.relu10 = paddle.nn.ReLU()
        self.conv2d12 = paddle.nn.Conv2D(
            stride=2, padding=1, out_channels=128, kernel_size=(3, 3), bias_attr=False, in_channels=128
        )
        self.batchnorm12 = paddle.nn.BatchNorm(is_test=True, num_channels=128, momentum=0.1)
        self.relu11 = paddle.nn.ReLU()
        self.conv2d13 = paddle.nn.Conv2D(out_channels=512, kernel_size=(1, 1), bias_attr=False, in_channels=128)
        self.batchnorm13 = paddle.nn.BatchNorm(is_test=True, num_channels=512, momentum=0.1)
        self.conv2d14 = paddle.nn.Conv2D(
            stride=2, out_channels=512, kernel_size=(1, 1), bias_attr=False, in_channels=256
        )
        self.batchnorm14 = paddle.nn.BatchNorm(is_test=True, num_channels=512, momentum=0.1)
        self.relu12 = paddle.nn.ReLU()
        self.conv2d15 = paddle.nn.Conv2D(out_channels=128, kernel_size=(1, 1), bias_attr=False, in_channels=512)
        self.batchnorm15 = paddle.nn.BatchNorm(is_test=True, num_channels=128, momentum=0.1)
        self.relu13 = paddle.nn.ReLU()
        self.conv2d16 = paddle.nn.Conv2D(
            padding=1, out_channels=128, kernel_size=(3, 3), bias_attr=False, in_channels=128
        )
        self.batchnorm16 = paddle.nn.BatchNorm(is_test=True, num_channels=128, momentum=0.1)
        self.relu14 = paddle.nn.ReLU()
        self.conv2d17 = paddle.nn.Conv2D(out_channels=512, kernel_size=(1, 1), bias_attr=False, in_channels=128)
        self.batchnorm17 = paddle.nn.BatchNorm(is_test=True, num_channels=512, momentum=0.1)
        self.relu15 = paddle.nn.ReLU()
        self.conv2d18 = paddle.nn.Conv2D(out_channels=128, kernel_size=(1, 1), bias_attr=False, in_channels=512)
        self.batchnorm18 = paddle.nn.BatchNorm(is_test=True, num_channels=128, momentum=0.1)
        self.relu16 = paddle.nn.ReLU()
        self.conv2d19 = paddle.nn.Conv2D(
            padding=1, out_channels=128, kernel_size=(3, 3), bias_attr=False, in_channels=128
        )
        self.batchnorm19 = paddle.nn.BatchNorm(is_test=True, num_channels=128, momentum=0.1)
        self.relu17 = paddle.nn.ReLU()
        self.conv2d20 = paddle.nn.Conv2D(out_channels=512, kernel_size=(1, 1), bias_attr=False, in_channels=128)
        self.batchnorm20 = paddle.nn.BatchNorm(is_test=True, num_channels=512, momentum=0.1)
        self.relu18 = paddle.nn.ReLU()
        self.conv2d21 = paddle.nn.Conv2D(out_channels=128, kernel_size=(1, 1), bias_attr=False, in_channels=512)
        self.batchnorm21 = paddle.nn.BatchNorm(is_test=True, num_channels=128, momentum=0.1)
        self.relu19 = paddle.nn.ReLU()
        self.conv2d22 = paddle.nn.Conv2D(
            padding=1, out_channels=128, kernel_size=(3, 3), bias_attr=False, in_channels=128
        )
        self.batchnorm22 = paddle.nn.BatchNorm(is_test=True, num_channels=128, momentum=0.1)
        self.relu20 = paddle.nn.ReLU()
        self.conv2d23 = paddle.nn.Conv2D(out_channels=512, kernel_size=(1, 1), bias_attr=False, in_channels=128)
        self.batchnorm23 = paddle.nn.BatchNorm(is_test=True, num_channels=512, momentum=0.1)
        self.relu21 = paddle.nn.ReLU()
        self.conv2d24 = paddle.nn.Conv2D(out_channels=256, kernel_size=(1, 1), bias_attr=False, in_channels=512)
        self.batchnorm24 = paddle.nn.BatchNorm(is_test=True, num_channels=256, momentum=0.1)
        self.relu22 = paddle.nn.ReLU()
        self.conv2d25 = paddle.nn.Conv2D(
            stride=2, padding=1, out_channels=256, kernel_size=(3, 3), bias_attr=False, in_channels=256
        )
        self.batchnorm25 = paddle.nn.BatchNorm(is_test=True, num_channels=256, momentum=0.1)
        self.relu23 = paddle.nn.ReLU()
        self.conv2d26 = paddle.nn.Conv2D(out_channels=1024, kernel_size=(1, 1), bias_attr=False, in_channels=256)
        self.batchnorm26 = paddle.nn.BatchNorm(is_test=True, num_channels=1024, momentum=0.1)
        self.conv2d27 = paddle.nn.Conv2D(
            stride=2, out_channels=1024, kernel_size=(1, 1), bias_attr=False, in_channels=512
        )
        self.batchnorm27 = paddle.nn.BatchNorm(is_test=True, num_channels=1024, momentum=0.1)
        self.relu24 = paddle.nn.ReLU()
        self.conv2d28 = paddle.nn.Conv2D(out_channels=256, kernel_size=(1, 1), bias_attr=False, in_channels=1024)
        self.batchnorm28 = paddle.nn.BatchNorm(is_test=True, num_channels=256, momentum=0.1)
        self.relu25 = paddle.nn.ReLU()
        self.conv2d29 = paddle.nn.Conv2D(
            padding=1, out_channels=256, kernel_size=(3, 3), bias_attr=False, in_channels=256
        )
        self.batchnorm29 = paddle.nn.BatchNorm(is_test=True, num_channels=256, momentum=0.1)
        self.relu26 = paddle.nn.ReLU()
        self.conv2d30 = paddle.nn.Conv2D(out_channels=1024, kernel_size=(1, 1), bias_attr=False, in_channels=256)
        self.batchnorm30 = paddle.nn.BatchNorm(is_test=True, num_channels=1024, momentum=0.1)
        self.relu27 = paddle.nn.ReLU()
        self.conv2d31 = paddle.nn.Conv2D(out_channels=256, kernel_size=(1, 1), bias_attr=False, in_channels=1024)
        self.batchnorm31 = paddle.nn.BatchNorm(is_test=True, num_channels=256, momentum=0.1)
        self.relu28 = paddle.nn.ReLU()
        self.conv2d32 = paddle.nn.Conv2D(
            padding=1, out_channels=256, kernel_size=(3, 3), bias_attr=False, in_channels=256
        )
        self.batchnorm32 = paddle.nn.BatchNorm(is_test=True, num_channels=256, momentum=0.1)
        self.relu29 = paddle.nn.ReLU()
        self.conv2d33 = paddle.nn.Conv2D(out_channels=1024, kernel_size=(1, 1), bias_attr=False, in_channels=256)
        self.batchnorm33 = paddle.nn.BatchNorm(is_test=True, num_channels=1024, momentum=0.1)
        self.relu30 = paddle.nn.ReLU()
        self.conv2d34 = paddle.nn.Conv2D(out_channels=256, kernel_size=(1, 1), bias_attr=False, in_channels=1024)
        self.batchnorm34 = paddle.nn.BatchNorm(is_test=True, num_channels=256, momentum=0.1)
        self.relu31 = paddle.nn.ReLU()
        self.conv2d35 = paddle.nn.Conv2D(
            padding=1, out_channels=256, kernel_size=(3, 3), bias_attr=False, in_channels=256
        )
        self.batchnorm35 = paddle.nn.BatchNorm(is_test=True, num_channels=256, momentum=0.1)
        self.relu32 = paddle.nn.ReLU()
        self.conv2d36 = paddle.nn.Conv2D(out_channels=1024, kernel_size=(1, 1), bias_attr=False, in_channels=256)
        self.batchnorm36 = paddle.nn.BatchNorm(is_test=True, num_channels=1024, momentum=0.1)
        self.relu33 = paddle.nn.ReLU()
        self.conv2d37 = paddle.nn.Conv2D(out_channels=256, kernel_size=(1, 1), bias_attr=False, in_channels=1024)
        self.batchnorm37 = paddle.nn.BatchNorm(is_test=True, num_channels=256, momentum=0.1)
        self.relu34 = paddle.nn.ReLU()
        self.conv2d38 = paddle.nn.Conv2D(
            padding=1, out_channels=256, kernel_size=(3, 3), bias_attr=False, in_channels=256
        )
        self.batchnorm38 = paddle.nn.BatchNorm(is_test=True, num_channels=256, momentum=0.1)
        self.relu35 = paddle.nn.ReLU()
        self.conv2d39 = paddle.nn.Conv2D(out_channels=1024, kernel_size=(1, 1), bias_attr=False, in_channels=256)
        self.batchnorm39 = paddle.nn.BatchNorm(is_test=True, num_channels=1024, momentum=0.1)
        self.relu36 = paddle.nn.ReLU()
        self.conv2d40 = paddle.nn.Conv2D(out_channels=256, kernel_size=(1, 1), bias_attr=False, in_channels=1024)
        self.batchnorm40 = paddle.nn.BatchNorm(is_test=True, num_channels=256, momentum=0.1)
        self.relu37 = paddle.nn.ReLU()
        self.conv2d41 = paddle.nn.Conv2D(
            padding=1, out_channels=256, kernel_size=(3, 3), bias_attr=False, in_channels=256
        )
        self.batchnorm41 = paddle.nn.BatchNorm(is_test=True, num_channels=256, momentum=0.1)
        self.relu38 = paddle.nn.ReLU()
        self.conv2d42 = paddle.nn.Conv2D(out_channels=1024, kernel_size=(1, 1), bias_attr=False, in_channels=256)
        self.batchnorm42 = paddle.nn.BatchNorm(is_test=True, num_channels=1024, momentum=0.1)
        self.relu39 = paddle.nn.ReLU()
        self.conv2d43 = paddle.nn.Conv2D(out_channels=512, kernel_size=(1, 1), bias_attr=False, in_channels=1024)
        self.batchnorm43 = paddle.nn.BatchNorm(is_test=True, num_channels=512, momentum=0.1)
        self.relu40 = paddle.nn.ReLU()
        self.conv2d44 = paddle.nn.Conv2D(
            stride=2, padding=1, out_channels=512, kernel_size=(3, 3), bias_attr=False, in_channels=512
        )
        self.batchnorm44 = paddle.nn.BatchNorm(is_test=True, num_channels=512, momentum=0.1)
        self.relu41 = paddle.nn.ReLU()
        self.conv2d45 = paddle.nn.Conv2D(out_channels=2048, kernel_size=(1, 1), bias_attr=False, in_channels=512)
        self.batchnorm45 = paddle.nn.BatchNorm(is_test=True, num_channels=2048, momentum=0.1)
        self.conv2d46 = paddle.nn.Conv2D(
            stride=2, out_channels=2048, kernel_size=(1, 1), bias_attr=False, in_channels=1024
        )
        self.batchnorm46 = paddle.nn.BatchNorm(is_test=True, num_channels=2048, momentum=0.1)
        self.relu42 = paddle.nn.ReLU()
        self.conv2d47 = paddle.nn.Conv2D(out_channels=512, kernel_size=(1, 1), bias_attr=False, in_channels=2048)
        self.batchnorm47 = paddle.nn.BatchNorm(is_test=True, num_channels=512, momentum=0.1)
        self.relu43 = paddle.nn.ReLU()
        self.conv2d48 = paddle.nn.Conv2D(
            padding=1, out_channels=512, kernel_size=(3, 3), bias_attr=False, in_channels=512
        )
        self.batchnorm48 = paddle.nn.BatchNorm(is_test=True, num_channels=512, momentum=0.1)
        self.relu44 = paddle.nn.ReLU()
        self.conv2d49 = paddle.nn.Conv2D(out_channels=2048, kernel_size=(1, 1), bias_attr=False, in_channels=512)
        self.batchnorm49 = paddle.nn.BatchNorm(is_test=True, num_channels=2048, momentum=0.1)
        self.relu45 = paddle.nn.ReLU()
        self.conv2d50 = paddle.nn.Conv2D(out_channels=512, kernel_size=(1, 1), bias_attr=False, in_channels=2048)
        self.batchnorm50 = paddle.nn.BatchNorm(is_test=True, num_channels=512, momentum=0.1)
        self.relu46 = paddle.nn.ReLU()
        self.conv2d51 = paddle.nn.Conv2D(
            padding=1, out_channels=512, kernel_size=(3, 3), bias_attr=False, in_channels=512
        )
        self.batchnorm51 = paddle.nn.BatchNorm(is_test=True, num_channels=512, momentum=0.1)
        self.relu47 = paddle.nn.ReLU()
        self.conv2d52 = paddle.nn.Conv2D(out_channels=2048, kernel_size=(1, 1), bias_attr=False, in_channels=512)
        self.batchnorm52 = paddle.nn.BatchNorm(is_test=True, num_channels=2048, momentum=0.1)
        self.relu48 = paddle.nn.ReLU()
        self.pool2d1 = paddle.nn.AdaptiveAvgPool2D(output_size=[1, 1])

    def forward(self, x0):
        x16 = self.conv2d0(x0)
        x25 = self.batchnorm0(x16)
        x26 = self.relu0(x25)
        x32 = self.pool2d0(x26)
        x54 = self.conv2d1(x32)
        x59 = self.batchnorm1(x54)
        x60 = self.relu1(x59)
        x66 = self.conv2d2(x60)
        x71 = self.batchnorm2(x66)
        x72 = self.relu2(x71)
        x78 = self.conv2d3(x72)
        x83 = self.batchnorm3(x78)
        x91 = self.conv2d4(x32)
        x96 = self.batchnorm4(x91)
        x97 = x83 + x96
        x98 = self.relu3(x97)
        x110 = self.conv2d5(x98)
        x115 = self.batchnorm5(x110)
        x116 = self.relu4(x115)
        x122 = self.conv2d6(x116)
        x127 = self.batchnorm6(x122)
        x128 = self.relu5(x127)
        x134 = self.conv2d7(x128)
        x139 = self.batchnorm7(x134)
        x140 = x139 + x98
        x141 = self.relu6(x140)
        x153 = self.conv2d8(x141)
        x158 = self.batchnorm8(x153)
        x159 = self.relu7(x158)
        x165 = self.conv2d9(x159)
        x170 = self.batchnorm9(x165)
        x171 = self.relu8(x170)
        x177 = self.conv2d10(x171)
        x182 = self.batchnorm10(x177)
        x183 = x182 + x141
        x184 = self.relu9(x183)
        x207 = self.conv2d11(x184)
        x212 = self.batchnorm11(x207)
        x213 = self.relu10(x212)
        x219 = self.conv2d12(x213)
        x224 = self.batchnorm12(x219)
        x225 = self.relu11(x224)
        x231 = self.conv2d13(x225)
        x236 = self.batchnorm13(x231)
        x244 = self.conv2d14(x184)
        x249 = self.batchnorm14(x244)
        x250 = x236 + x249
        x251 = self.relu12(x250)
        x263 = self.conv2d15(x251)
        x268 = self.batchnorm15(x263)
        x269 = self.relu13(x268)
        x275 = self.conv2d16(x269)
        x280 = self.batchnorm16(x275)
        x281 = self.relu14(x280)
        x287 = self.conv2d17(x281)
        x292 = self.batchnorm17(x287)
        x293 = x292 + x251
        x294 = self.relu15(x293)
        x306 = self.conv2d18(x294)
        x311 = self.batchnorm18(x306)
        x312 = self.relu16(x311)
        x318 = self.conv2d19(x312)
        x323 = self.batchnorm19(x318)
        x324 = self.relu17(x323)
        x330 = self.conv2d20(x324)
        x335 = self.batchnorm20(x330)
        x336 = x335 + x294
        x337 = self.relu18(x336)
        x349 = self.conv2d21(x337)
        x354 = self.batchnorm21(x349)
        x355 = self.relu19(x354)
        x361 = self.conv2d22(x355)
        x366 = self.batchnorm22(x361)
        x367 = self.relu20(x366)
        x373 = self.conv2d23(x367)
        x378 = self.batchnorm23(x373)
        x379 = x378 + x337
        x380 = self.relu21(x379)
        x405 = self.conv2d24(x380)
        x410 = self.batchnorm24(x405)
        x411 = self.relu22(x410)
        x417 = self.conv2d25(x411)
        x422 = self.batchnorm25(x417)
        x423 = self.relu23(x422)
        x429 = self.conv2d26(x423)
        x434 = self.batchnorm26(x429)
        x442 = self.conv2d27(x380)
        x447 = self.batchnorm27(x442)
        x448 = x434 + x447
        x449 = self.relu24(x448)
        x461 = self.conv2d28(x449)
        x466 = self.batchnorm28(x461)
        x467 = self.relu25(x466)
        x473 = self.conv2d29(x467)
        x478 = self.batchnorm29(x473)
        x479 = self.relu26(x478)
        x485 = self.conv2d30(x479)
        x490 = self.batchnorm30(x485)
        x491 = x490 + x449
        x492 = self.relu27(x491)
        x504 = self.conv2d31(x492)
        x509 = self.batchnorm31(x504)
        x510 = self.relu28(x509)
        x516 = self.conv2d32(x510)
        x521 = self.batchnorm32(x516)
        x522 = self.relu29(x521)
        x528 = self.conv2d33(x522)
        x533 = self.batchnorm33(x528)
        x534 = x533 + x492
        x535 = self.relu30(x534)
        x547 = self.conv2d34(x535)
        x552 = self.batchnorm34(x547)
        x553 = self.relu31(x552)
        x559 = self.conv2d35(x553)
        x564 = self.batchnorm35(x559)
        x565 = self.relu32(x564)
        x571 = self.conv2d36(x565)
        x576 = self.batchnorm36(x571)
        x577 = x576 + x535
        x578 = self.relu33(x577)
        x590 = self.conv2d37(x578)
        x595 = self.batchnorm37(x590)
        x596 = self.relu34(x595)
        x602 = self.conv2d38(x596)
        x607 = self.batchnorm38(x602)
        x608 = self.relu35(x607)
        x614 = self.conv2d39(x608)
        x619 = self.batchnorm39(x614)
        x620 = x619 + x578
        x621 = self.relu36(x620)
        x633 = self.conv2d40(x621)
        x638 = self.batchnorm40(x633)
        x639 = self.relu37(x638)
        x645 = self.conv2d41(x639)
        x650 = self.batchnorm41(x645)
        x651 = self.relu38(x650)
        x657 = self.conv2d42(x651)
        x662 = self.batchnorm42(x657)
        x663 = x662 + x621
        x664 = self.relu39(x663)
        x686 = self.conv2d43(x664)
        x691 = self.batchnorm43(x686)
        x692 = self.relu40(x691)
        x698 = self.conv2d44(x692)
        x703 = self.batchnorm44(x698)
        x704 = self.relu41(x703)
        x710 = self.conv2d45(x704)
        x715 = self.batchnorm45(x710)
        x723 = self.conv2d46(x664)
        x728 = self.batchnorm46(x723)
        x729 = x715 + x728
        x730 = self.relu42(x729)
        x742 = self.conv2d47(x730)
        x747 = self.batchnorm47(x742)
        x748 = self.relu43(x747)
        x754 = self.conv2d48(x748)
        x759 = self.batchnorm48(x754)
        x760 = self.relu44(x759)
        x766 = self.conv2d49(x760)
        x771 = self.batchnorm49(x766)
        x772 = x771 + x730
        x773 = self.relu45(x772)
        x785 = self.conv2d50(x773)
        x790 = self.batchnorm50(x785)
        x791 = self.relu46(x790)
        x797 = self.conv2d51(x791)
        x802 = self.batchnorm51(x797)
        x803 = self.relu47(x802)
        x809 = self.conv2d52(x803)
        x814 = self.batchnorm52(x809)
        x815 = x814 + x773
        x816 = self.relu48(x815)
        x818 = self.pool2d1(x816)
        return x818


def main(x0):
    # There are 1 inputs.
    # x0: shape-[1, 3, 256, 256], type-float32.
    paddle.disable_static()
    params = paddle.load(r"/workspace/X2Paddle/test_benchmark/PyTorch/ARNIQA/pd_model/model.pdparams")
    model = Sequential()
    model.set_dict(params, use_structured_name=True)
    model.eval()
    out = model(x0)
    return out
