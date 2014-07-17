/* BEGIN COPYRIGHT
 *
 * This file is part of Noted.
 *
 * Copyright ©2011, 2012, Lancaster Logic Response Limited.
 *
 * Noted is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 2 of the License, or
 * (at your option) any later version.
 *
 * Noted is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with Noted.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <QtGui/QPainter>

#include "GraphParameters.h"
#include "Grapher.h"

using namespace std;
using namespace lb;

void Grapher::init(QPainter* _p, std::pair<float, float> _xRange, std::pair<float, float> _yRange, std::function<std::string(float)> _xLabel, std::function<std::string(float)> _yLabel, std::function<std::string(float, float)> _pLabel, int _leftGutter, int _bottomGutter)
{
	fontPixelSize = QFontInfo(QFont("Ubuntu", 10)).pixelSize();

	if (_leftGutter)
		_leftGutter = max<int>(_leftGutter, fontPixelSize * 2);
	if (_bottomGutter)
		_bottomGutter = max<int>(_bottomGutter, fontPixelSize * 1.25);

	QRect a(_leftGutter, 0, _p->viewport().width() - _leftGutter, _p->viewport().height() - _bottomGutter);
	init(_p, _xRange, _yRange, _xLabel, _yLabel, _pLabel, a);
}

bool Grapher::drawAxes(bool _x, bool _y) const
{
	int w = active.width();
	int h = active.height();
	int l = active.left();
	int r = active.right();
	int t = active.top();
	int b = active.bottom();

	p->setFont(QFont("Ubuntu", 10));
	p->fillRect(p->viewport(), qRgb(255, 255, 255));

	static const int c_markLength = 2;
	static const int c_markSpacing = 2;
	static const int c_xSpacing = fontPixelSize * 3;
	static const int c_ySpacing = fontPixelSize * 1.25;

	if (w < c_xSpacing || h < c_ySpacing || !p->viewport().contains(active))
		return false;

	if (_y)
	{
		GraphParameters<float> yParams(yRange, h / c_ySpacing, 1.f);
		float dy = fabs(yRange.second - yRange.first);
		if (dy > .001)
			for (float f = yParams.from; f < yParams.to; f += yParams.incr)
			{
				int y = b - h * (f - yParams.from) / dy;
				if (yParams.isMajor(f))
				{
					p->setPen(QColor(208, 208, 208));
					p->drawLine(l - c_markLength, y, r, y);
					if (l > p->viewport().left())
					{
						p->setPen(QColor(144, 144, 144));
						p->drawText(QRect(0, y - c_ySpacing / 2, l - c_markLength - c_markSpacing, c_ySpacing), Qt::AlignRight|Qt::AlignVCenter, QString::fromStdString(yLabel(round(f * 100000) / 100000)));
					}
				}
				else
				{
					p->setPen(QColor(236, 236, 236));
					p->drawLine(l, y, r, y);
				}
			}
		p->setPen(QColor(192,192,192));
		p->drawLine(l - c_markSpacing, b, r, b);
	}

	if (_x)
	{
		GraphParameters<float> xParams(xRange, w / c_xSpacing, 1.f);
		float dx = fabs(xRange.second - xRange.first);
		for (float f = xParams.from; f < xParams.to; f += xParams.incr)
		{
			int x = l + w * (f - xParams.from) / dx;
			if (xParams.isMajor(f))
			{
				p->setPen(QColor(208, 208, 208));
				p->drawLine(x, t, x, b + c_markLength);
				if (b < p->viewport().bottom())
				{
					p->setPen(QColor(144, 144, 144));
					p->drawText(QRect(x - c_xSpacing / 2, b + c_markLength + c_markSpacing, c_xSpacing, p->viewport().height() - (b + c_markLength + c_markSpacing)), Qt::AlignHCenter|Qt::AlignTop, QString::fromStdString(xLabel(f)));
				}
			}
			else
			{
				p->setPen(QColor(236, 236, 236));
				p->drawLine(x, t, x, b);
			}
		}
	}

	p->setClipRect(active);
	return true;
}

void Grapher::drawLineGraph(vector<float> const& _data, QColor _color, QBrush const& _fillToZero, float _width) const
{
	int s = _data.size();
	QPoint l;
	for (int i = 0; i < s; ++i)
	{
		int zy = yP(0.f);
		QPoint h(xTP(i), yTP(_data[i]));
		if (i)
		{
			if (_fillToZero != Qt::NoBrush)
			{
				p->setPen(Qt::NoPen);
				p->setBrush(_fillToZero);
				p->drawPolygon(QPolygon(QVector<QPoint>() << QPoint(h.x(), zy) << h << l << QPoint(l.x(), zy)));
			}
			p->setPen(QPen(_color, _width));
			p->drawLine(QLine(l, h));
		}
		l = h;
	}
}

void Grapher::ruleY(float _x, QColor _color, float _width) const
{
	p->setPen(QPen(_color, _width));
	p->drawLine(xTP(_x), active.top(), xTP(_x), active.bottom());
}

void Grapher::drawLineGraph(std::function<float(float)> const& _f, QColor _color, QBrush const& _fillToZero, float _width) const
{
	QPoint l;
	for (int x = active.left(); x < active.right(); x += 2)
	{
		int zy = yP(0.f);
		QPoint h(x, yTP(_f(xRU(x))));
		if (x != active.left())
		{
			if (_fillToZero != Qt::NoBrush)
			{
				p->setPen(Qt::NoPen);
				p->setBrush(_fillToZero);
				p->drawPolygon(QPolygon(QVector<QPoint>() << QPoint(h.x(), zy) << h << l << QPoint(l.x(), zy)));
			}
			p->setPen(QPen(_color, _width));
			p->drawLine(QLine(l, h));
		}
		l = h;
	}
}

void Grapher::labelYOrderedPoints(map<float, float> const& _data, int _maxCount, float _minFactor) const
{
	int ly = active.top() + 6;
	int pc = 0;
	if (_data.empty())
		return;
	float smallestAllowed = prev(_data.end())->first * _minFactor;
	for (auto peaki = _data.rbegin(); peaki != _data.rend(); ++peaki)
		if ((peaki->first > smallestAllowed || _minFactor == 0) && pc < _maxCount)
		{
			auto peak = *peaki;
			int x = xTP(peak.second);
			int y = yTP(peak.first);
			p->setPen(QColor::fromHsvF(float(pc) / _maxCount, 1.f, 0.5f, 0.5f));
			p->drawEllipse(QPoint(x, y), 4, 4);
			p->drawLine(x, y - 4, x, ly + 6);
			QString f = QString::fromStdString(pLabel(xT(peak.second), yT(peak.first)));
			int fw = p->fontMetrics().width(f);
			p->drawLine(x + 16 + fw + 2, ly + 6, x, ly + 6);
			p->setPen(QColor::fromHsvF(0, 0.f, .35f));
			p->fillRect(QRect(x+12, ly-6, fw + 8, 12), QBrush(QColor(255, 255, 255, 160)));
			p->drawText(QRect(x+16, ly-6, 160, 12), Qt::AlignVCenter, f);
			ly += 14;
			++pc;
		}
}
