//debugData => contain all debug states.
//statesList => ListView

var currentSelectedState = null;
var jumpStartingPoint = null;
var debugData = null;
var codeMap = null;

function init(data)
{
	jumpOutBackAction.enabled(false);
	jumpIntoBackAction.enabled(false);
	jumpIntoForwardAction.enabled(false);
	jumpOutForwardAction.enabled(false);

	if (data === null) {
		statesList.model.clear();
		statesSlider.maximumValue = 0;
		statesSlider.value = 0;
		currentSelectedState = null;
		jumpStartingPoint = null;
		debugData = null;
		return;
	}

	debugData = data;
	statesSlider.maximumValue = data.states.length - 1;
	statesSlider.value = 0;
	currentSelectedState = 0;
	setupInstructions(currentSelectedState);
	select(currentSelectedState);
}

function setupInstructions(stateIndex) {

	var instructions = debugData.states[stateIndex].code.instructions;
	codeMap = {};
	statesList.model.clear();
	for (var i = 0; i < instructions.length; i++) {
		statesList.model.append(instructions[i]);
		codeMap[instructions[i].processIndex] = i;
	}
}

function moveSelection(incr)
{
	var prevState = currentSelectedState;
	if (currentSelectedState + incr >= 0)
	{
		if (currentSelectedState + incr < debugData.states.length)
			select(currentSelectedState + incr);
		statesSlider.value = currentSelectedState;
	}
}

function select(stateIndex)
{
	if (debugData.states[stateIndex].codeIndex !== debugData.states[currentSelectedState].codeIndex)
		setupInstructions(stateIndex);
	currentSelectedState = stateIndex;
	var codeLine = codeStr(stateIndex);
	var state = debugData.states[stateIndex];
	highlightSelection(codeLine);
	completeCtxInformation(state);

	if (state.instruction === "CALL" || state.instruction === "CREATE")
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
	var state = debugData.states[stateIndex];
	return codeMap[state.curPC];
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
	gasSpent.update(debugData.states[0].gas.subtract(state.gas).value());

	stack.listModel = state.debugStack;
	storage.listModel = state.debugStorage;
	memoryDump.listModel = state.debugMemory;
	callDataDump.listModel = state.debugCallData;
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
	var state = debugData.states[currentSelectedState];
	if (state.instruction === "CALL" || state.instruction === "CREATE")
	{
		for (var k = currentSelectedState; k > 0; k--)
		{
			var line = codeMap[debugData.states[k].curPC];
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
	var state = debugData.states[currentSelectedState];
	if (state.instruction === "CALL" || state.instruction === "CREATE")
	{
		for (var k = currentSelectedState; k < debugData.states.length; k++)
		{
			var line = codeMap[debugData.states[k].curPC];
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
	var state = debugData.states[currentSelectedState];
	if (state.instruction === "CALL" || state.instruction === "CREATE")
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
	select(value);
}
