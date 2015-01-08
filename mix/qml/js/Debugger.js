//humanReadableExecutionCode => contain human readable code.
//debugStates => contain all debug states.
//bytesCodeMapping => mapping between humanReadableExecutionCode and bytesCode.
//statesList => ListView

var currentSelectedState = null;
var jumpStartingPoint = null;
function init()
{
	statesSlider.maximumValue = debugStates.length - 1;
	statesList.model = humanReadableExecutionCode;
	currentSelectedState = 0;
	select(currentSelectedState);
	//displayReturnValue();

	jumpoutbackaction.state = "disabled";
	jumpintobackaction.state = "disabled";
	jumpintoforwardaction.state = "disabled"
	jumpoutforwardaction.state = "disabled"

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
			//endOfDebug();
		}
		statesSlider.value = currentSelectedState;
	}
}

function select(stateIndex)
{
	var codeLine = codeStr(stateIndex);
	var state = debugStates[stateIndex];
	highlightSelection(codeLine);
	currentSelectedState = stateIndex;
	completeCtxInformation(state);
	//levelList.model = state.levels;
	//levelList.update();

	if (state.instruction === "JUMP")
		jumpintoforwardaction.state = "";
	else
		jumpintoforwardaction.state = "disabled";

	if (state.instruction === "JUMPDEST")
		jumpintobackaction.state = "";
	else
		jumpintobackaction.state = "disabled";
}

function codeStr(stateIndex)
{
	var state = debugStates[stateIndex];
	return bytesCodeMapping.getValue(state.curPC);
}

function highlightSelection(index)
{
	statesList.currentIndex = index;
}

function completeCtxInformation(state)
{
	basicInfo.currentStep = state.step;
	basicInfo.mem = state.newMemSize + " " + qsTr("words");
	basicInfo.stepCost = state.gasCost;
	basicInfo.gasSpent = debugStates[0].gas - state.gas;
	// This is available in all editors.
	stack.listModel = state.debugStack;
	storage.listModel = state.debugStorage;
	memoryDump.listModel = state.debugMemory;
	callDataDump.listModel = state.debugCallData;
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

function stepOutBack()
{
	if (jumpStartingPoint != null)
	{
		select(jumpStartingPoint);
		jumpStartingPoint = null;
		jumpoutbackaction.state = "disabled";
		jumpoutforwardaction.state = "disabled";
	}
}

function stepIntoBack()
{
	moveSelection(-1);
}

function stepOverBack()
{
	var state = debugStates[currentSelectedState];
	if (state.instruction === "JUMPDEST")
	{
		for (var k = currentSelectedState; k > 0; k--)
		{
			var line = bytesCodeMapping.getValue(debugStates[k].curPC);
			if (line === statesList.currentIndex - 2)
			{
				select(k);
				break;
			}
		}
	}
	else
		moveSelection(-1);
}

function stepOverForward()
{
	var state = debugStates[currentSelectedState];
	if (state.instruction === "JUMP")
	{
		for (var k = currentSelectedState; k < debugStates.length; k++)
		{
			var line = bytesCodeMapping.getValue(debugStates[k].curPC);
			if (line === statesList.currentIndex + 2)
			{
				select(k);
				break;
			}
		}
	}
	else
		moveSelection(1);
}

function stepIntoForward()
{
	var state = debugStates[currentSelectedState];
	if (state.instruction === "JUMP")
	{
		jumpStartingPoint = currentSelectedState;
		moveSelection(1);
		jumpoutbackaction.state = "";
		jumpoutforwardaction.state = "";
	}
}

function stepOutForward()
{
	if (jumpStartingPoint != null)
	{
		stepOutBack();
		stepOverForward();
		jumpoutbackaction.state = "disabled";
		jumpoutforwardaction.state = "disabled";
	}
}

function jumpTo(value)
{
	currentSelectedState = value;
	select(currentSelectedState);
}
