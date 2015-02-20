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

#pragma once

#include <map>
#include <vector>
#include <string>
#include <utility>
#include <functional>

#include <QtGui/QBrush>
#include <QtCore/QRect>

class QPainter;

namespace lb
{

class Grapher
{
public:
	Grapher(): p(0) {}
	void init(QPainter* _p, std::pair<float, float> _xRange, std::pair<float, float> _yRange, std::function<std::string(float _f)> _xLabel, std::function<std::string(float _f)> _yLabel, std::function<std::string(float, float)> _pLabel, int _leftGutter = 30, int _bottomGutter = 16);
	void init(QPainter* _p, std::pair<float, float> _xRange, std::pair<float, float> _yRange, std::function<std::string(float _f)> _xLabel, std::function<std::string(float _f)> _yLabel, std::function<std::string(float, float)> _pLabel, QRect _active)
	{
		p = _p;
		active = _active;
		xRange = _xRange;
		yRange = _yRange;
		dx = xRange.second - xRange.first;
		dy = yRange.second - yRange.first;
		xLabel = _xLabel;
		yLabel = _yLabel;
		pLabel = _pLabel;
	}

	void setDataTransform(float _xM, float _xC, float _yM, float _yC)
	{
		xM = _xM;
		xC = _xC;
		yM = _yM;
		yC = _yC;
	}
	void setDataTransform(float _xM, float _xC)
	{
		xM = _xM;
		xC = _xC;
		yM = 1.f;
		yC = 0.f;
	}
	void resetDataTransform() { xM = yM = 1.f; xC = yC = 0.f; }

	bool drawAxes(bool _x = true, bool _y = true) const;
	void drawLineGraph(std::vector<float> const& _data, QColor _color = QColor(128, 128, 128), QBrush const& _fillToZero = Qt::NoBrush, float _width = 0.f) const;
	void drawLineGraph(std::function<float(float)> const& _f, QColor _color = QColor(128, 128, 128), QBrush const& _fillToZero = Qt::NoBrush, float _width = 0.f) const;
	void ruleX(float _y, QColor _color = QColor(128, 128, 128), float _width = 0.f) const;
	void ruleY(float _x, QColor _color = QColor(128, 128, 128), float _width = 0.f) const;
	void labelYOrderedPoints(std::map<float, float> const& _translatedData, int _maxCount = 20, float _minFactor = .01f) const;

protected:
	QPainter* p = nullptr;
	QRect active;
	std::pair<float, float> xRange;
	std::pair<float, float> yRange;

	float xM = 0;
	float xC = 0;
	float yM = 0;
	float yC = 0;

	float dx = 0;
	float dy = 0;

	std::function<std::string(float _f)> xLabel;
	std::function<std::string(float _f)> yLabel;
	std::function<std::string(float _x, float _y)> pLabel;

	float fontPixelSize = 0;

	// Translate from raw indexed data into x/y graph units. Only relevant for indexed data.
	float xT(float _dataIndex) const { return _dataIndex * xM + xC; }
	float yT(float _dataValue) const { return _dataValue * yM + yC; }
	// Translate from x/y graph units to widget pixels.
	int xP(float _xUnits) const { return active.left() + (_xUnits - xRange.first) / dx * active.width(); }
	int yP(float _yUnits) const { return active.bottom() - (_yUnits - yRange.first) / dy * active.height(); }
	QPoint P(float _x, float _y) const { return QPoint(xP(_x), yP(_y)); }
	// Translate direcly from raw indexed data to widget pixels.
	int xTP(float _dataIndex) const { return active.left() + (xT(_dataIndex) - xRange.first) / dx * active.width(); }
	int yTP(float _dataValue) const { return active.bottom() - (yT(_dataValue) - yRange.first) / dy * active.height(); }
	// Translate back from pixels into graph units.
	float xU(int _xPixels) const { return float(_xPixels - active.left()) / active.width() * dx + xRange.first; }
	// Translate back from graph units into raw data index.
	float xR(float _xUnits) const { return (_xUnits - xC) / xM; }
	// Translate directly from pixels into raw data index. xRU(xTP(X)) == X
	float xRU(int _xPixels) const { return xR(xU(_xPixels)); }
};

}
