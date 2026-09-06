import unittest
import json
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from src.graph.supervisor_agent import (
    query_site_telemetry,
    compile_work_permit,
    adjudicate_incident_packet,
    generate_daily_toolbox_talk,
    get_active_learning_queue,
    update_exclusion_corridor,
    build_supervisor_agent,
    SupervisorAgentState
)


class TestSupervisorAgent(unittest.TestCase):
    def test_query_site_telemetry_tool(self):
        res = query_site_telemetry.invoke({"scenario_id": "scenario_worker_in_excavator_blind_spot"})
        data = json.loads(res)
        self.assertIn("scenario_id", data)
        self.assertIn("total_frames", data)
        self.assertIn("critical_near_miss", data)

    def test_compile_work_permit_tool(self):
        permit = "Shift PTW #412: Foundation excavation at Pier B4. Heavy articulated equipment active. Spotter present."
        res = compile_work_permit.invoke({
            "permit_text": permit,
            "site_id": "SITE-01",
            "shift_date": "2026-09-07"
        })
        data = json.loads(res)
        self.assertIn("tasks_extracted", data)
        self.assertGreater(data["tasks_extracted"], 0)
        self.assertIn("SUCCESS_DISPATCHED", data["dispatch_status"])

    def test_adjudicate_incident_packet_tool(self):
        res = adjudicate_incident_packet.invoke({
            "scenario_id": "scenario_worker_in_excavator_blind_spot",
            "event_id": "INC-TEST-009"
        })
        data = json.loads(res)
        self.assertEqual(data["event_id"], "INC-TEST-009")
        self.assertIn("verdict", data)
        self.assertIn("verdict", data["verdict"])
        self.assertEqual(data["verdict"]["verdict"], "TRUE_POSITIVE")

    def test_generate_daily_toolbox_talk_tool(self):
        res = generate_daily_toolbox_talk.invoke({
            "site_id": "SITE-NORTH",
            "focus_hazard": "Excavator Swing Trajectory"
        })
        data = json.loads(res)
        self.assertEqual(data["site_id"], "SITE-NORTH")
        self.assertIn("osha_standard", data)
        self.assertIn("field_protocols", data)
        self.assertGreater(len(data["field_protocols"]), 0)

    def test_get_active_learning_queue_tool(self):
        res = get_active_learning_queue.invoke({})
        data = json.loads(res)
        self.assertIn("total_curated", data)
        self.assertIn("high_priority_count", data)

    def test_update_exclusion_corridor_tool(self):
        res = update_exclusion_corridor.invoke({
            "zone_id": "ZONE_EXCAVATOR_RADIUS",
            "radius_meters": 6.5,
            "site_id": "SITE-01"
        })
        data = json.loads(res)
        self.assertEqual(data["zone_id"], "ZONE_EXCAVATOR_RADIUS")
        self.assertEqual(data["new_radius_meters"], 6.5)

    def test_full_supervisor_agent_toolbox_talk_loop(self):
        agent = build_supervisor_agent()
        initial_state: SupervisorAgentState = {
            "messages": [HumanMessage(content="Generate the daily toolbox briefing for SITE-01 focus on heavy machinery")],
            "site_id": "SITE-01",
            "current_step": 0
        }
        res = agent.invoke(initial_state)
        messages = res.get("messages", [])
        self.assertGreater(len(messages), 1)
        # Should contain tool execution
        has_tool_message = any(isinstance(m, ToolMessage) for m in messages)
        self.assertTrue(has_tool_message)
        # Final message should be an AIMessage
        self.assertIsInstance(messages[-1], AIMessage)
        self.assertIn("Toolbox", messages[-1].content)

    def test_full_supervisor_agent_telemetry_loop(self):
        agent = build_supervisor_agent()
        initial_state: SupervisorAgentState = {
            "messages": [HumanMessage(content="Check live telemetry and status for scenario_worker_in_excavator_blind_spot")],
            "site_id": "SITE-01",
            "current_step": 0
        }
        res = agent.invoke(initial_state)
        messages = res.get("messages", [])
        self.assertGreater(len(messages), 1)
        has_tool_message = any(isinstance(m, ToolMessage) for m in messages)
        self.assertTrue(has_tool_message)
        self.assertIsInstance(messages[-1], AIMessage)
        self.assertIn("Telemetry", messages[-1].content)


if __name__ == "__main__":
    unittest.main()
