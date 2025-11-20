import re
from typing import Optional, List, Dict, Tuple, Union

import autogen
import openai
from autogen import UserProxyAgent, Agent, OpenAIWrapper

from defense.explicit_detector.explicit_defense_arch import ExplicitMultiAgentDefense, DefenseAssistantAgent, DefenseGroupChat
from defense.utility import load_defense_prompt
from evaluator.evaluate_helper import evaluate_defense_with_response

# =============================================================================
# CONFIGURABLE CONFIDENCE THRESHOLDS - MODIFY THESE TO EXPERIMENT
# =============================================================================
CONFIDENCE_CONFIG = {
    # Early exit thresholds (0.0 - 1.0)
    "intention_analyzer_threshold": 0.8,   # Lower = more early exits, higher = fewer early exits
    "prompt_analyzer_threshold": 0.8,      # Lower = more early exits, higher = fewer early exits
    
    # Enable/disable early exits entirely
    "enable_early_exit": True,              # Set to False to disable early exits completely
    
    # Debug mode - prints routing decisions
    "debug_routing": True,                  # Set to True to see routing decisions in console
}

# Quick presets - uncomment one of these to use instead of manual config above
# CONFIDENCE_CONFIG = {"intention_analyzer_threshold": 0.9, "prompt_analyzer_threshold": 0.9, "enable_early_exit": True, "debug_routing": False}  # Conservative
# CONFIDENCE_CONFIG = {"intention_analyzer_threshold": 0.7, "prompt_analyzer_threshold": 0.7, "enable_early_exit": True, "debug_routing": False}  # Aggressive
# CONFIDENCE_CONFIG = {"intention_analyzer_threshold": 0.8, "prompt_analyzer_threshold": 0.8, "enable_early_exit": False, "debug_routing": False}  # No early exit
# =============================================================================


class ConfidenceAwareAgent(DefenseAssistantAgent):
    """Agent that extracts confidence scores from its responses"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.last_confidence = 0.5
        self.last_verdict = "UNKNOWN"
    
    def extract_confidence_and_verdict(self, response_text: str) -> Tuple[str, float]:
        """Extract verdict and confidence from agent response"""
        # Look for confidence score patterns
        confidence_patterns = [
            r"Confidence:\s*([0-9]*\.?[0-9]+)",
            r"confidence:\s*([0-9]*\.?[0-9]+)",
            r"Confidence Score:\s*([0-9]*\.?[0-9]+)",
        ]
        
        confidence = 0.5  # Default moderate confidence
        for pattern in confidence_patterns:
            match = re.search(pattern, response_text, re.IGNORECASE)
            if match:
                try:
                    confidence = float(match.group(1))
                    if confidence > 1.0:  # Handle percentage format (e.g., 85%)
                        confidence = confidence / 100.0
                    break
                except ValueError:
                    continue
        
        # Extract verdict
        verdict = "UNKNOWN"
        if "VALID" in response_text.upper():
            verdict = "ACCEPT"
        elif "INVALID" in response_text.upper():
            verdict = "REJECT"
        
        self.last_confidence = confidence
        self.last_verdict = verdict
        
        if CONFIDENCE_CONFIG.get("debug_routing", False):
            print(f"[{self.name}] Verdict: {verdict}, Confidence: {confidence:.2f}")
        
        return verdict, confidence

    def generate_oai_reply(
            self,
            messages: Optional[List[Dict]] = None,
            sender: Optional[Agent] = None,
            config: Optional[OpenAIWrapper] = None,
    ) -> Tuple[bool, Union[str, Dict, None]]:
        # Get standard response
        final, response = super().generate_oai_reply(messages, sender, config)
        
        if response and isinstance(response, str):
            # Extract confidence and verdict for routing decisions
            self.extract_confidence_and_verdict(response)
            
        return final, response


class DynamicRouterCoordinator(DefenseAssistantAgent):
    """Coordinator that implements confidence-based early exit routing"""
    
    def __init__(self, **kwargs):
        kwargs["name"] = "Coordinator"
        self.strategy_name = "explicit_3_agent"  # Reuse existing prompts
        super().__init__(**kwargs)
        self.defense_prompt = load_defense_prompt()
        self.register_reply([Agent, None], DynamicRouterCoordinator.generate_dynamic_route_reply)
        
        # Performance tracking
        self.early_exits = 0
        self.total_decisions = 0
        self.early_exit_result = None  # Store early exit results

    def should_early_exit(self, agent_name: str, verdict: str, confidence: float) -> bool:
        """Determine if we should exit early based on configuration"""
        
        # Check if early exit is enabled
        if not CONFIDENCE_CONFIG.get("enable_early_exit", True):
            if CONFIDENCE_CONFIG.get("debug_routing", False):
                print(f"[Router] Early exit disabled in config")
            return False
            
        # Only exit early on REJECT verdicts
        if verdict != "REJECT":
            if CONFIDENCE_CONFIG.get("debug_routing", False):
                print(f"[Router] No early exit - verdict is {verdict}, need REJECT")
            return False
            
        # Check confidence against thresholds
        if agent_name == "IntentionAnalyzer":
            threshold = CONFIDENCE_CONFIG.get("intention_analyzer_threshold", 0.8)
            will_exit = confidence >= threshold
            if CONFIDENCE_CONFIG.get("debug_routing", False):
                print(f"[Router] Intention Analyzer: confidence {confidence:.2f} >= threshold {threshold} = {will_exit}")
            return will_exit
        elif agent_name == "OriginalPromptAnalyzer":
            threshold = CONFIDENCE_CONFIG.get("prompt_analyzer_threshold", 0.8)
            will_exit = confidence >= threshold
            if CONFIDENCE_CONFIG.get("debug_routing", False):
                print(f"[Router] Prompt Analyzer: confidence {confidence:.2f} >= threshold {threshold} = {will_exit}")
            return will_exit
        
        return False

    def generate_dynamic_route_reply(
            self,
            messages: Optional[List[Dict]] = None,
            sender: Optional[Agent] = None,
            config: Optional[OpenAIWrapper] = None,
    ) -> Tuple[bool, Union[str, Dict, None]]:
        
        last_agent_name = self._oai_messages[sender][-1]['name']
        self.total_decisions += 1
        
        if last_agent_name == "TaskInputAgent":
            # Start with Intention Analyzer
            if CONFIDENCE_CONFIG.get("debug_routing", False):
                print(f"[Router] Starting analysis pipeline")
            self.early_exit_result = None  # Reset early exit result
            return True, self.defense_prompt[self.strategy_name]["1_user"]
            
        elif last_agent_name == "IntentionAnalyzer":
            # Check for early exit opportunity
            if hasattr(sender, 'last_confidence') and hasattr(sender, 'last_verdict'):
                verdict = sender.last_verdict
                confidence = sender.last_confidence
                
                if CONFIDENCE_CONFIG.get("debug_routing", False):
                    print(f"[Router] Checking early exit for IntentionAnalyzer: verdict={verdict}, confidence={confidence:.2f}")
                
                if self.should_early_exit("IntentionAnalyzer", verdict, confidence):
                    self.early_exits += 1
                    if CONFIDENCE_CONFIG.get("debug_routing", False):
                        print(f"[Router] EARLY EXIT: Intention Analyzer rejected with {confidence:.2f} confidence")
                    # Store the early exit result for retrieval in __call__
                    self.early_exit_result = "INVALID"
                    return True, "Next: TaskInputAgent"  # Signal termination
            
            # Continue to Prompt Analyzer
            if CONFIDENCE_CONFIG.get("debug_routing", False):
                print(f"[Router] Continuing to Prompt Analyzer")
            return True, self.defense_prompt[self.strategy_name]["2_user"]
            
        elif last_agent_name == "OriginalPromptAnalyzer":
            # Check for early exit opportunity  
            if hasattr(sender, 'last_confidence') and hasattr(sender, 'last_verdict'):
                verdict = sender.last_verdict
                confidence = sender.last_confidence
                
                if CONFIDENCE_CONFIG.get("debug_routing", False):
                    print(f"[Router] Checking early exit for OriginalPromptAnalyzer: verdict={verdict}, confidence={confidence:.2f}")
                
                if self.should_early_exit("OriginalPromptAnalyzer", verdict, confidence):
                    self.early_exits += 1
                    if CONFIDENCE_CONFIG.get("debug_routing", False):
                        print(f"[Router] EARLY EXIT: Prompt Analyzer rejected with {confidence:.2f} confidence")
                    # Store the early exit result for retrieval in __call__
                    self.early_exit_result = "INVALID"
                    return True, "Next: TaskInputAgent"  # Signal termination
            
            # Continue to Judge for final decision
            if CONFIDENCE_CONFIG.get("debug_routing", False):
                print(f"[Router] Continuing to Judge for final decision")
            system_input = re.findall(r"--SYSTEM INPUT START--\n((.|\n)*)--SYSTEM INPUT END--",
                                    self._oai_messages[sender][0]['content'])[0][0]
            return True, self.defense_prompt[self.strategy_name]["3_user"].replace("[INSERT INPUT HERE]", system_input)
            
        elif last_agent_name == "Judge":
            if CONFIDENCE_CONFIG.get("debug_routing", False):
                print(f"[Router] Analysis complete via Judge")
            return True, "Next: TaskInputAgent"
            
        else:
            raise ValueError(f"Unknown agent name: {last_agent_name}")

    def get_performance_stats(self):
        """Get routing performance statistics"""
        early_exit_rate = self.early_exits / max(1, self.total_decisions)
        return {
            "early_exits": self.early_exits,
            "total_decisions": self.total_decisions,
            "early_exit_rate": early_exit_rate,
            "config": CONFIDENCE_CONFIG.copy()
        }


class DynamicRouterDetector:
    """Drop-in replacement for AutoGenDetectorThreeAgency with configurable dynamic routing"""
    
    def __init__(self, config_list):
        self.defense_prompt = load_defense_prompt()
        self.strategy_name = "explicit_3_agent"  # Reuse existing prompts
        self.config = {
            "cache_seed": 123,
            "config_list": config_list,
            "timeout": 300,
        }
        self.init_agents()

    def init_agents(self):
        """Initialize agents with confidence-aware prompts"""
        self.user_input_agent = UserProxyAgent(name="TaskInputAgent",
                                               human_input_mode="NEVER",
                                               is_termination_msg=lambda x: True)
        
        self.coordinator = DynamicRouterCoordinator(llm_config=self.config)
        
        # Enhance existing prompts with confidence instructions
        intention_prompt = self.defense_prompt[self.strategy_name]["1_system"] + \
                          "\n\nIMPORTANT: You must end your response with 'Confidence: X.XX' where X.XX is your confidence level (0.00-1.00) in your assessment."
        
        prompt_prompt = self.defense_prompt[self.strategy_name]["2_system"] + \
                       "\n\nIMPORTANT: You must end your response with 'Confidence: X.XX' where X.XX is your confidence level (0.00-1.00) in your assessment."
        
        self.agent_athena = ConfidenceAwareAgent(
            name="IntentionAnalyzer",
            system_message=intention_prompt,
            llm_config=self.config,
        )
        self.agent_hermes = ConfidenceAwareAgent(
            name="OriginalPromptAnalyzer", 
            system_message=prompt_prompt,
            llm_config=self.config,
        )
        self.agent_themis = DefenseAssistantAgent(
            name="Judge",
            system_message=self.defense_prompt[self.strategy_name]["3_system"],
            llm_config=self.config
        )
        
        groupchat = DefenseGroupChat(
            agents=[self.user_input_agent, self.coordinator, self.agent_athena, self.agent_hermes, self.agent_themis],
            messages=[], max_round=20)
        self.manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=self.config)

    def clear_history(self):
        """Clear conversation history"""
        self.user_input_agent.clear_history()
        self.agent_athena.clear_history()
        self.agent_hermes.clear_history()
        self.coordinator.clear_history()
        self.agent_themis.clear_history()

    def __call__(self, content_text, user_prompt=None):
        """Same interface as AutoGenDetectorThreeAgency"""
        self.clear_history()
        try:
            self.user_input_agent.initiate_chat(self.manager, 
                                               message=self.defense_prompt[self.strategy_name]["0_user"].
                                               replace("[INSERT INPUT HERE]", content_text), silent=True)
            
            # Check if we have an early exit result
            if hasattr(self.coordinator, 'early_exit_result') and self.coordinator.early_exit_result:
                llm_judgement_output = self.coordinator.early_exit_result
                if CONFIDENCE_CONFIG.get("debug_routing", False):
                    print(f"[Main] Using early exit result: {llm_judgement_output}")
            else:
                # Get the judge's final decision
                llm_judgement_output = self.manager._oai_messages[self.agent_themis][-2]['content']
                if CONFIDENCE_CONFIG.get("debug_routing", False):
                    print(f"[Main] Using Judge result: {llm_judgement_output}")

            success = True
            if "INVALID" in llm_judgement_output:
                valid = False
            elif "VALID" in llm_judgement_output:
                valid = True
            else:
                valid = False
                success = False
                
        except openai.BadRequestError as e:
            valid = False
            success = True
            llm_judgement_output = "ERROR" + str(e)

        return valid, success, llm_judgement_output

    def agency_execute(self, llm_output):
        """Same interface as AutoGenDetectorThreeAgency"""
        valid, success, llm_output = self(llm_output)
        return llm_output

    def get_performance_stats(self):
        """Get routing performance statistics"""
        return self.coordinator.get_performance_stats()


if __name__ == "__main__":
    # Example usage
    print("Dynamic Router Configuration:")
    for key, value in CONFIDENCE_CONFIG.items():
        print(f"  {key}: {value}")
    print()
    
    # Test with existing evaluation framework
    evaluate_defense_with_response(task_agency=DynamicRouterDetector,
                                   defense_agency=ExplicitMultiAgentDefense,
                                   model_name="mistral-7b-instruct",
                                   chat_file="data/harmful_output/gpt-35-turbo-1106/attack-dan.json",
                                   defense_output_name="/tmp/dynamic_router_test.json",
                                   parallel=False)