//debugData => contain all debug states.
//statesList => ListView

var currentSelectedState = null;
var currentDisplayedState = null;
var debugData = null;
var codeMap = null;

function init(data)
{
	jumpOutBackAction.enabled(false);
	jumpIntoBackAction.enabled(false);
	jumpIntoForwardAction.enabled(false);
	jumpOutForwardAction.enabled(false);
	jumpOverBackAction.enabled(false);
	jumpOverForwardAction.enabled(false);

	if (data === null) {
		statesList.model.clear();
		statesSlider.maximumValue = 0;
		statesSlider.value = 0;
		currentSelectedState = null;
		currentDisplayedState = null;
		debugData = null;
		return;
	}

	debugData = data;
	currentSelectedState = 0;
	currentDisplayedState = 0;
	setupInstructions(currentSelectedState);
	setupCallData(currentSelectedState);
	statesSlider.maximumValue = data.states.length - 1;
	statesSlider.value = 0;
	select(currentSelectedState);
}

function setupInstructions(stateIndex)
{
	var instructions = debugData.states[stateIndex].code.instructions;
	codeMap = {};
	statesList.model.clear();
	for (var i = 0; i < instructions.length; i++) {
		statesList.model.append(instructions[i]);
		codeMap[instructions[i].processIndex] = i;
	}
	callDataDump.listModel = debugData.states[stateIndex].callData.items;
}

function setupCallData(stateIndex)
{
	callDataDump.listModel = debugData.states[stateIndex].callData.items;
}

function moveSelection(incr)
{
	var prevState = currentSelectedState;
	if (currentSelectedState + incr >= 0)
	{
		if (currentSelectedState + incr < debugData.states.length)
			select(currentSelectedState + incr);
	}
}

function display(stateIndex)
{
	if (stateIndex < 0)
		stateIndex = 0;
	if (stateIndex >= debugData.states.length)
		stateIndex = debugData.state.length - 1;
	if (debugData.states[stateIndex].codeIndex !== debugData.states[currentDisplayedState].codeIndex)
		setupInstructions(stateIndex);
	if (debugData.states[stateIndex].dataIndex !== debugData.states[currentDisplayedState].dataIndex)
		setupCallData(stateIndex);
	var codeLine = codeStr(stateIndex);
	var state = debugData.states[stateIndex];
	highlightSelection(codeLine);
	completeCtxInformation(state);
	currentDisplayedState = stateIndex;
}

function displayFrame(frameIndex)
{
	var state = debugData.states[currentSelectedState];
	if (frameIndex === 0)
		display(currentSelectedState);
	else
		display(state.levels[frameIndex - 1]);
}

function select(stateIndex)
{
	display(stateIndex);
	currentSelectedState = stateIndex;
	var state = debugData.states[stateIndex];
	statesSlider.value = stateIndex;
	jumpIntoForwardAction.enabled(stateIndex < debugData.states.length - 1)
	jumpIntoBackAction.enabled(stateIndex > 0);
	jumpOverForwardAction.enabled(stateIndex < debugData.states.length - 1);
	jumpOverBackAction.enabled(stateIndex > 0);
	jumpOutBackAction.enabled(state.levels.length > 1);
	jumpOutForwardAction.enabled(state.levels.length > 1);

	var callStackData = [];
	for (var l = 0; l < state.levels.length; l++) {
		var address = debugData.states[state.levels[l] + 1].address;
		callStackData.push(address);
	}
	callStackData.push(debugData.states[0].address);
	callStack.listModel = callStackData;
}

function codeStr(stateIndex)
{
	var state = debugData.states[stateIndex];
	return codeMap[state.curPC];
}

function highlightSelection(index)
{
	statesList.positionViewAtRow(index, ListView.Center);
	statesList.selection.clear();
	statesList.selection.select(index);
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
}

function isCallInstruction(index)
{
	var state = debugData.states[index];
	return state.instruction === "CALL" || state.instruction === "CREATE";
}

function isReturnInstruction(index)
{
	var state = debugData.states[index];
	return state.instruction === "RETURN"
}

function stepIntoBack()
{
	moveSelection(-1);
}

function stepOverBack()
{
	if (currentSelectedState > 0 && isReturnInstruction(currentSelectedState - 1))
		stepOutBack();
	else
		moveSelection(-1);
}

function stepOverForward()
{
	if (isCallInstruction(currentSelectedState))
		stepOutForward();
	else
		moveSelection(1);
}

function stepIntoForward()
{
	moveSelection(1);
}

function stepOutBack()
{
	var i = currentSelectedState - 1;
	var depth = 0;
	while (--i >= 0)
		if (isCallInstruction(i))
			if (depth == 0)
				break;
			else depth--;
		else if (isReturnInstruction(i))
			depth++;
	select(i);
}

function stepOutForward()
{
	var i = currentSelectedState;
	var depth = 0;
	while (++i < debugData.states.length)
		if (isReturnInstruction(i))
			if (depth == 0)
				break;
			else
				depth--;
		else if (isCallInstruction(i))
			depth++;
	select(i + 1);
}

function jumpTo(value)
{
	select(value);
}
