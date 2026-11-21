import pytest
from pydantic import ValidationError
from processlens.models.process import ProcessModel, Step
from processlens.core.llm import LLMClient


def test_valid_linear_process():
    step1 = Step(
        id="step_1",
        name="Receive invoice",
        actor="Accounts Payable",
        next_step_ids=["step_2"],
        confidence=0.98,
    )
    step2 = Step(
        id="step_2",
        name="Approve invoice",
        actor="Finance Manager",
        next_step_ids=["step_3"],
        confidence=0.95,
    )
    step3 = Step(
        id="step_3",
        name="Disburse payment",
        actor="Treasury",
        next_step_ids=[],
        confidence=0.99,
    )

    model = ProcessModel(
        title="Invoice Processing",
        version="1.0",
        steps=[step1, step2, step3],
    )
    assert model.title == "Invoice Processing"
    assert len(model.steps) == 3
    assert "Accounts Payable" in model.actors
    assert "Finance Manager" in model.actors
    assert "Treasury" in model.actors


def test_single_step_process():
    step = Step(
        id="s1",
        name="Send ping",
        actor="System",
        next_step_ids=[],
    )
    model = ProcessModel(title="Ping Process", steps=[step])
    assert len(model.steps) == 1
    assert model.actors == ["System"]


def test_invalid_step_reference():
    step1 = Step(
        id="step_1",
        name="Submit Form",
        actor="Employee",
        next_step_ids=["non_existent_step"],
    )
    with pytest.raises(ValidationError) as exc_info:
        ProcessModel(title="Form Submission", steps=[step1])
    assert "references unknown next step" in str(exc_info.value)


def test_duplicate_step_ids():
    step1 = Step(
        id="dup_id",
        name="Review 1",
        actor="Reviewer",
        next_step_ids=["step_2"],
    )
    step2 = Step(
        id="dup_id",
        name="Review 2",
        actor="Reviewer",
        next_step_ids=[],
    )
    with pytest.raises(ValidationError) as exc_info:
        ProcessModel(title="Duplicate Test", steps=[step1, step2])
    assert "Duplicate step IDs" in str(exc_info.value)


def test_isolated_orphan_step():
    step1 = Step(
        id="s1",
        name="Step 1",
        actor="User",
        next_step_ids=["s2"],
    )
    step2 = Step(
        id="s2",
        name="Step 2",
        actor="User",
        next_step_ids=[],
    )
    orphan = Step(
        id="s_orphan",
        name="Orphan Step",
        actor="User",
        next_step_ids=[],
    )
    with pytest.raises(ValidationError) as exc_info:
        ProcessModel(title="Orphan Test", steps=[step1, step2, orphan])
    assert "Orphan step detected" in str(exc_info.value)


def test_disconnected_component_orphan():
    step1 = Step(
        id="s1",
        name="Step 1",
        actor="User",
        next_step_ids=["s2"],
    )
    step2 = Step(
        id="s2",
        name="Step 2",
        actor="User",
        next_step_ids=[],
    )
    step3 = Step(
        id="s3",
        name="Step 3",
        actor="Admin",
        next_step_ids=["s4"],
    )
    step4 = Step(
        id="s4",
        name="Step 4",
        actor="Admin",
        next_step_ids=[],
    )
    with pytest.raises(ValidationError) as exc_info:
        ProcessModel(title="Disconnected Subgraphs", steps=[step1, step2, step3, step4])
    assert "Orphan steps detected in disconnected component" in str(exc_info.value)


def test_unreachable_step_orphan():
    step1 = Step(
        id="s1",
        name="Step 1",
        actor="User",
        next_step_ids=["s2"],
    )
    step2 = Step(
        id="s2",
        name="Step 2",
        actor="User",
        next_step_ids=[],
    )
    step3 = Step(
        id="s3",
        name="Step 3",
        actor="User",
        next_step_ids=["s2"],
    )
    step4 = Step(
        id="s4",
        name="Step 4",
        actor="User",
        next_step_ids=["s3"],
    )
    model = ProcessModel(title="Multiple Valid Entries", steps=[step1, step2, step3, step4])
    assert len(model.steps) == 4


def test_unmarked_cycle_rejected():
    step1 = Step(
        id="s1",
        name="Draft document",
        actor="Author",
        next_step_ids=["s2"],
    )
    step2 = Step(
        id="s2",
        name="Review document",
        actor="Manager",
        next_step_ids=["s1"],
        is_loop=False,
    )
    with pytest.raises(ValidationError) as exc_info:
        ProcessModel(title="Cycle Test", steps=[step1, step2])
    assert "Cycle detected without explicit loop marker" in str(exc_info.value)


def test_marked_cycle_allowed():
    step1 = Step(
        id="s1",
        name="Draft document",
        actor="Author",
        next_step_ids=["s2"],
    )
    step2 = Step(
        id="s2",
        name="Review document",
        actor="Manager",
        next_step_ids=["s1", "s3"],
        is_decision=True,
        conditions=["Rejected", "Approved"],
        is_loop=True,
    )
    step3 = Step(
        id="s3",
        name="Publish document",
        actor="Publisher",
        next_step_ids=[],
    )
    model = ProcessModel(title="Approved Loop Process", steps=[step1, step2, step3])
    assert len(model.steps) == 3


def test_unmarked_self_loop_rejected():
    step1 = Step(
        id="s1",
        name="Self repeating",
        actor="Bot",
        next_step_ids=["s1"],
        is_loop=False,
    )
    with pytest.raises(ValidationError) as exc_info:
        ProcessModel(title="Self Loop Test", steps=[step1])
    assert "Cycle detected without explicit loop marker" in str(exc_info.value)


def test_marked_self_loop_allowed():
    step1 = Step(
        id="s1",
        name="Self repeating",
        actor="Bot",
        next_step_ids=["s1"],
        is_loop=True,
    )
    model = ProcessModel(title="Self Loop Allowed", steps=[step1])
    assert len(model.steps) == 1


def test_confidence_validation():
    with pytest.raises(ValidationError):
        Step(
            id="s1",
            name="Invalid Confidence High",
            actor="User",
            confidence=1.5,
        )

    with pytest.raises(ValidationError):
        Step(
            id="s2",
            name="Invalid Confidence Low",
            actor="User",
            confidence=-0.1,
        )


def test_mocked_llm_structured_generation():
    mock_json = """
    {
        "title": "Mock Onboarding Process",
        "version": "1.0",
        "actors": ["HR", "IT"],
        "steps": [
            {
                "id": "step_1",
                "name": "Prepare contract",
                "description": "HR prepares the employment contract",
                "actor": "HR",
                "inputs": ["Candidate details"],
                "outputs": ["Signed contract"],
                "next_step_ids": ["step_2"],
                "is_decision": false,
                "conditions": [],
                "source_chunk_ids": ["chunk_0"],
                "confidence": 0.99,
                "is_loop": false
            },
            {
                "id": "step_2",
                "name": "Provision laptop",
                "description": "IT provisions developer laptop",
                "actor": "IT",
                "inputs": ["Signed contract"],
                "outputs": ["Configured laptop"],
                "next_step_ids": [],
                "is_decision": false,
                "conditions": [],
                "source_chunk_ids": ["chunk_1"],
                "confidence": 0.96,
                "is_loop": false
            }
        ]
    }
    """
    client = LLMClient(mock_response=mock_json)
    result = client.generate_structured(
        prompt="Extract onboarding",
        response_model=ProcessModel,
    )
    assert result.title == "Mock Onboarding Process"
    assert len(result.steps) == 2
    assert result.steps[0].actor == "HR"
    assert result.steps[1].actor == "IT"
