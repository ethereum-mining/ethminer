//humanReadableExecutionCode => contain human readable code.
//debugStates => contain all debug states.
//bytesCodeMapping => mapping between humanReadableExecutionCode and bytesCode.
//statesList => ListView

var currentSelectedState = null;
var jumpStartingPoint = null;
function init()
{
	if (debugStates === undefined)
		return;

	statesSlider.maximumValue = debugStates.length - 1;
	statesSlider.value = 0;
	statesList.model = humanReadableExecutionCode;
	currentSelectedState = 0;
	select(currentSelectedState);

	jumpOutBackAction.enabled(false);
	jumpIntoBackAction.enabled(false);
	jumpIntoForwardAction.enabled(false);
	jumpOutForwardAction.enabled(false);
}

function moveSelection(incr)
{
	if (currentSelectedState + incr >= 0)
	{
		if (currentSelectedState + incr < debugStates.length)
			select(currentSelectedState + incr);
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

	if (state.instruction === "JUMP")
		jumpIntoForwardAction.enabled(true);
	else
		jumpIntoForwardAction.enabled(false);

	if (state.instruction === "JUMPDEST")
		jumpIntoBackAction.enabled(true);
	else
		jumpIntoBackAction.enabled(false);
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
	currentStep.update(state.step);
	mem.update(state.newMemSize.value() + " " + qsTr("words"));
	stepCost.update(state.gasCost.value());
	gasSpent.update(debugStates[0].gas.subtract(state.gas).value());

	stack.listModel = state.debugStack;
	storage.listModel = state.debugStorage;
	memoryDump.listModel = state.debugMemory;
	callDataDump.listModel = state.debugCallData;
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
		jumpOutBackAction.enabled(false);
		jumpOutForwardAction.enabled(false);
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
		jumpOutBackAction.enabled(true);
		jumpOutForwardAction.enabled(true);
	}
}

function stepOutForward()
{
	if (jumpStartingPoint != null)
	{
		stepOutBack();
		stepOverForward();
		jumpOutBackAction.enabled(false);
		jumpOutForwardAction.enabled(false);
	}
}

function jumpTo(value)
{
	currentSelectedState = value;
	select(currentSelectedState);
}
