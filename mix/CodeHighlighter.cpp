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
/** @file CodeHighlighter.cpp
 * @author Arkadiy Paronyan arkadiy@ethdev.com
 * @date 2015
 * Ethereum IDE client.
 */

#include <algorithm>
#include <QRegularExpression>
#include <QTextDocument>
#include <QTextBlock>
#include <QTextLayout>
#include <libsolidity/ASTVisitor.h>
#include <libsolidity/AST.h>
#include <libsolidity/Scanner.h>
#include <libsolidity/Exceptions.h>
#include "CodeHighlighter.h"

using namespace dev::mix;

CodeHighlighterSettings::CodeHighlighterSettings()
{
	backgroundColor = QColor(0x00, 0x2b, 0x36);
	foregroundColor = QColor(0xee, 0xe8, 0xd5);
	formats[Keyword].setForeground(QColor(0x93, 0xa1, 0xa1));
	formats[Comment].setForeground(QColor(0x85, 0x99, 0x00));
	formats[StringLiteral].setForeground(QColor(0xdc, 0x32, 0x2f));
	formats[NumLiteral].setForeground(foregroundColor);
	formats[Import].setForeground(QColor(0x6c, 0x71, 0xc4));
	formats[CompilationError].setUnderlineColor(Qt::red);
	formats[CompilationError].setUnderlineStyle(QTextCharFormat::SingleUnderline);
}

namespace
{
	using namespace dev::solidity;
	class HighlightVisitor: public ASTConstVisitor
	{
	public:
		HighlightVisitor(CodeHighlighter::Formats* _formats) { m_formats = _formats; }
	private:
		CodeHighlighter::Formats* m_formats;

		virtual bool visit(ImportDirective const& _node)
		{
			m_formats->push_back(CodeHighlighter::FormatRange(CodeHighlighterSettings::Import, _node.getLocation()));
			return true;
		}
	};
}

CodeHighlighter::FormatRange::FormatRange(CodeHighlighterSettings::Token _t, dev::solidity::Location const& _location):
	token(_t), start(_location.start), length(_location.end - _location.start)
{}

void CodeHighlighter::processSource(std::string const& _source)
{
	processComments(_source);
	solidity::CharStream stream(_source);
	solidity::Scanner scanner(stream);
	solidity::Token::Value token = scanner.getCurrentToken();
	while (token != Token::EOS)
	{
		if ((token >= Token::Break && token < Token::TypesEnd) ||
				token == Token::In || token == Token::Delete || token == Token::NullLiteral || token == Token::TrueLiteral || token == Token::FalseLiteral)
			m_formats.push_back(FormatRange(CodeHighlighterSettings::Keyword, scanner.getCurrentLocation()));
		else if (token == Token::StringLiteral)
			m_formats.push_back(FormatRange(CodeHighlighterSettings::StringLiteral, scanner.getCurrentLocation()));
		else if (token == Token::CommentLiteral)
			m_formats.push_back(FormatRange(CodeHighlighterSettings::Comment, scanner.getCurrentLocation()));
		else if (token == Token::Number)
			m_formats.push_back(FormatRange(CodeHighlighterSettings::NumLiteral, scanner.getCurrentLocation()));

		token = scanner.next();
	}
	std::sort(m_formats.begin(), m_formats.end());
}

void CodeHighlighter::processAST(dev::solidity::ASTNode const& _ast)
{
	HighlightVisitor visitor(&m_formats);
	_ast.accept(visitor);

	std::sort(m_formats.begin(), m_formats.end());
}

void CodeHighlighter::processError(dev::Exception const& _exception)
{
	Location const* location = boost::get_error_info<errinfo_sourceLocation>(_exception);
	if (location)
		m_formats.push_back(FormatRange(CodeHighlighterSettings::CompilationError, *location));
}

void CodeHighlighter::processComments(std::string const& _source)
{
	unsigned i = 0;
	unsigned size = _source.size();
	if (size == 0)
		return;
	while (i < size - 1)
	{
		if (_source[i] == '/' && _source[i + 1] == '/')
		{
			//add single line comment
			int start = i;
			i += 2;
			while (_source[i] != '\n' && i < size)
				++i;
			m_formats.push_back(FormatRange(CodeHighlighterSettings::Comment, start, i - start));
		}
		else if (_source[i] == '/' && _source[i + 1] == '*')
		{
			//add multiline comment
			int start = i;
			i += 2;
			while ((_source[i] != '/' || _source[i - 1] != '*') && i < size)
				++i;
			m_formats.push_back(FormatRange(CodeHighlighterSettings::Comment, start, i - start + 1));
		}
		++i;
	}
}

void CodeHighlighter::updateFormatting(QTextDocument* _document, CodeHighlighterSettings const& _settings)
{
	QTextBlock block = _document->firstBlock();
	QList<QTextLayout::FormatRange> ranges;

	Formats::const_iterator format = m_formats.begin();
	while (true)
	{
		while ((format == m_formats.end() || (block.position() + block.length() <= format->start)) && block.isValid())
		{
			auto layout = block.layout();
			layout->clearAdditionalFormats();
			layout->setAdditionalFormats(ranges);
			_document->markContentsDirty(block.position(), block.length());
			block = block.next();
			ranges.clear();
		}
		if (!block.isValid())
			break;

		int intersectionStart = std::max(format->start, block.position());
		int intersectionLength = std::min(format->start + format->length, block.position() + block.length()) - intersectionStart;
		if (intersectionLength > 0)
		{
			QTextLayout::FormatRange range;
			range.format = _settings.formats[format->token];
			range.start = format->start - block.position();
			range.length = format->length;
			ranges.append(range);
		}
		++format;
	}
}
