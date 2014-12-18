//humanReadableExecutionCode => contain human readable code.
//debugStates => contain all debug states.
//bytesCodeMapping => mapping between humanReadableExecutionCode and bytesCode.
//statesList => ListView

var currentSelectedState = null;
function init()
{
	currentSelectedState = 0;
	select(currentSelectedState);
	displayReturnValue();
}

function moveSelection(incr)
{
	if (currentSelectedState + incr >= 0)
	{
		if (currentSelectedState + incr < debugStates.length)
		{
			select(currentSelectedState + incr);
		}
		else
		{
			endOfDebug();
		}
	}
}

function select(stateIndex)
{
	var state = debugStates[stateIndex];
	var codeStr = bytesCodeMapping.getValue(state.curPC);
	highlightSelection(codeStr);
	currentSelectedState = stateIndex;
	completeCtxInformation(state);
	levelList.model = state.levels;
	levelList.update();
}

function highlightSelection(index)
{
	statesList.currentIndex = index;
}

function completeCtxInformation(state)
{
	debugStackTxt.text = state.debugStack;
	debugStorageTxt.text = state.debugStorage;
	debugMemoryTxt.text = state.debugMemory;
	debugCallDataTxt.text = state.debugCallData;
	headerInfoLabel.text = state.headerInfo
}

function endOfDebug()
{
	var state = debugStates[debugStates.length - 1];
	debugStorageTxt.text = "";
	debugCallDataTxt.text = "";
	debugStackTxt.text = "";
	debugMemoryTxt.text = state.endOfDebug;
	headerInfoLabel.text = "EXIT  |  GAS: " + state.gasLeft;
}

function displayReturnValue()
{
	headerReturnList.model = contractCallReturnParameters;
	headerReturnList.update();
}
