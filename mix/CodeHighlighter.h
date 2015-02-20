/*
    This file is part of cpp-ethereum.

    cpp-ethereum is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    cpp-ethereum is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with cpp-ethereum.  If not, see <http://www.gnu.org/licenses/>.
*/
/** @file CodeHighlighter.h
 * @author Arkadiy Paronyan arkadiy@ethdev.com
 * @date 2015
 * Ethereum IDE client.
 */

#pragma once

#include <vector>
#include <QString>
#include <QTextCharFormat>

class QTextDocument;

namespace dev
{

struct Exception;

namespace solidity
{
	class ASTNode;
	struct Location;
}

namespace mix
{

/// Code highligting settings
class CodeHighlighterSettings
{
public:
	enum Token
	{
		Import,
		Keyword,
		Comment,
		StringLiteral,
		NumLiteral,
		CompilationError,
		Size, //this must be kept last
	};

	CodeHighlighterSettings();
	///Format for each token
	QTextCharFormat formats[Size];
	///Background color
	QColor backgroundColor;
	///Foreground color
	QColor foregroundColor;
};

/// Code highlighting engine class
class CodeHighlighter
{
public:
	/// Formatting range
	struct FormatRange
	{
		FormatRange(CodeHighlighterSettings::Token _t, int _start, int _length): token(_t), start(_start), length(_length) {}
		FormatRange(CodeHighlighterSettings::Token _t, solidity::Location const& _location);
		bool operator<(FormatRange const& _other) const { return start < _other.start || (start == _other.start && length < _other.length); }

		CodeHighlighterSettings::Token token;
		int start;
		int length;
	};
	using Formats = std::vector<FormatRange>; // Sorted by start position

public:
	/// Collect highligting information by lexing the source
	void processSource(std::string const& _source);
	/// Collect additional highligting information from AST
	void processAST(solidity::ASTNode const& _ast);
	/// Collect highlighting information from compilation exception
	void processError(dev::Exception const& _exception);

	/// Apply formatting for a text document
	/// @todo Remove this once editor is reworked
	void updateFormatting(QTextDocument* _document, CodeHighlighterSettings const& _settings);

private:
	/// Collect highligting information by paring for comments
	/// @todo Support this in solidity?
	void processComments(std::string const& _source);

private:
	Formats m_formats;
};

}

}
