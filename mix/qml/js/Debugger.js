//debugData => contain all debug states.
//statesList => ListView

var currentSelectedState = null;
var currentDisplayedState = null;
var debugData = null;
var locations = [];
var locationMap = {};
var breakpoints = {};

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
		locations = [];
		locationMap = {};
		return;
	}

	debugData = data;
	currentSelectedState = 0;
	currentDisplayedState = 0;
	setupInstructions(currentSelectedState);
	setupCallData(currentSelectedState);
	initLocations();
	initSlider();
	selectState(currentSelectedState);
}

function updateMode()
{
	initSlider();
}

function initLocations()
{
	locations = [];
	if (debugData.states.length === 0)
		return;

	var nullLocation = { start: -1, end: -1, documentId: "", state: 0 };
	var prevLocation = nullLocation;

	for (var i = 0; i < debugData.states.length - 1; i++) {
		var code = debugData.states[i].code;
		var location = code.documentId ? debugData.states[i].solidity : nullLocation;
		if (location.start !== prevLocation.start || location.end !== prevLocation.end || code.documentId !== prevLocation.documentId)
		{
			prevLocation = { start: location.start, end: location.end, documentId: code.documentId, state: i };
			locations.push(prevLocation);
		}
		locationMap[i] = locations.length - 1;
	}
	locations.push({ start: -1, end: -1, documentId: code.documentId, state: i });

	locationMap[debugData.states.length - 1] = locations.length - 1;
}

function setBreakpoints(bp)
{
	breakpoints = bp;
}

function srcMode()
{
	return !assemblyMode && locations.length;
}

function initSlider()
{
	if (!debugData)
		statesSlider.maximumValue = 0;
	else if (srcMode()) {
		statesSlider.maximumValue = locations.length - 1;
	} else {
		statesSlider.maximumValue = debugData.states.length - 1;
	}
	statesSlider.value = 0;
}

function setupInstructions(stateIndex)
{
	var instructions = debugData.states[stateIndex].code.instructions;
	statesList.model.clear();
	for (var i = 0; i < instructions.length; i++)
		statesList.model.append(instructions[i]);

	callDataDump.listModel = debugData.states[stateIndex].callData.items;
}

function setupCallData(stateIndex)
{
	callDataDump.listModel = debugData.states[stateIndex].callData.items;
}

function moveSelection(incr)
{
	if (srcMode()) {
		var locationIndex = locationMap[currentSelectedState];
		if (locationIndex + incr >= 0 && locationIndex + incr < locations.length)
			selectState(locations[locationIndex + incr].state);
	} else {
		if (currentSelectedState + incr >= 0 && currentSelectedState + incr < debugData.states.length)
			selectState(currentSelectedState + incr);
	}
}

function display(stateIndex)
{
	if (stateIndex < 0)
		stateIndex = 0;
	if (stateIndex >= debugData.states.length)
		stateIndex = debugData.states.length - 1;
	if (debugData.states[stateIndex].codeIndex !== debugData.states[currentDisplayedState].codeIndex)
		setupInstructions(stateIndex);
	if (debugData.states[stateIndex].dataIndex !== debugData.states[currentDisplayedState].dataIndex)
		setupCallData(stateIndex);
	var state = debugData.states[stateIndex];
	var codeLine = state.instructionIndex;
	highlightSelection(codeLine);
	completeCtxInformation(state);
	currentDisplayedState = stateIndex;
	var docId = debugData.states[stateIndex].code.documentId;
	if (docId)
		debugExecuteLocation(docId, debugData.states[stateIndex].solidity);
}

function displayFrame(frameIndex)
{
	var state = debugData.states[currentSelectedState];
	if (frameIndex === 0)
		display(currentSelectedState);
	else
		display(state.levels[frameIndex - 1]);
}

function select(index)
{
	if (srcMode())
		selectState(locations[index].state);
	else
		selectState(index);
}

function selectState(stateIndex)
{
	display(stateIndex);
	currentSelectedState = stateIndex;
	var state = debugData.states[stateIndex];
	jumpIntoForwardAction.enabled(stateIndex < debugData.states.length - 1)
	jumpIntoBackAction.enabled(stateIndex > 0);
	jumpOverForwardAction.enabled(stateIndex < debugData.states.length - 1);
	jumpOverBackAction.enabled(stateIndex > 0);
	jumpOutBackAction.enabled(state.levels.length > 1);
	jumpOutForwardAction.enabled(state.levels.length > 1);
	runForwardAction.enabled(stateIndex < debugData.states.length - 1)
	runBackAction.enabled(stateIndex > 0);

	var callStackData = [];
	for (var l = 0; l < state.levels.length; l++) {
		var address = debugData.states[state.levels[l] + 1].code.address;
		callStackData.push(address);
	}
	callStackData.push(debugData.states[0].code.address);
	callStack.listModel = callStackData;
	if (srcMode())
		statesSlider.value = locationMap[stateIndex];
	else
		statesSlider.value = stateIndex;
}

function highlightSelection(index)
{
	if (statesList.visible)
		statesList.positionViewAtRow(index, ListView.Visible);
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
	if (state.solidity) {
		solLocals.setData(state.solidity.locals.variables, state.solidity.locals.values);
		solStorage.setData(state.solidity.storage.variables, state.solidity.storage.values);
		solCallStack.listModel = state.solidity.callStack;
	} else {
		solLocals.setData([], {});
		solStorage.setData([], {});
		solCallStack.listModel = [];
	}
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

function locationsIntersect(l1, l2)
{
	return l1.start <= l2.end && l1.end >= l2.start;
}

function breakpointHit(i)
{
	var bpLocations = breakpoints[debugData.states[i].code.documentId];
	if (bpLocations) {
		var location = debugData.states[i].solidity;
		if (location.start >= 0 && location.end >= location.start)
			for (var b = 0; b < bpLocations.length; b++)
				if (locationsIntersect(location, bpLocations[b]))
					return true;
	}
	return false;
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

function runBack()
{
	var i = currentSelectedState - 1;
	while (i > 0 && !breakpointHit(i)) {
		--i;
	}
	selectState(i);
}

function runForward()
{
	var i = currentSelectedState + 1;
	while (i < debugData.states.length - 1 && !breakpointHit(i)) {
		++i;
	}
	selectState(i);
}

function stepOutBack()
{
	var i = currentSelectedState - 1;
	var depth = 0;
	while (--i >= 0) {
		if (breakpointHit(i))
			break;
		if (isCallInstruction(i))
			if (depth == 0)
				break;
			else depth--;
		else if (isReturnInstruction(i))
			depth++;
	}
	selectState(i);
}

function stepOutForward()
{
	var i = currentSelectedState;
	var depth = 0;
	while (++i < debugData.states.length) {
		if (breakpointHit(i))
			break;
		if (isReturnInstruction(i))
			if (depth == 0)
				break;
			else
				depth--;
		else if (isCallInstruction(i))
			depth++;
	}
	selectState(i + 1);
}

function jumpTo(value)
{
	select(value);
}
