#!/usr/bin/env python3
"""
Data Contracts Validation Script
==================================
Validates YAML data contracts for MLOps Assignment 2

Author: Laksh Mendpara (B23CS1037)
Course: ML-DL-Ops (CSL7120)
"""

import yaml
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime


class ContractValidator:
    """Validates data contracts against assignment requirements"""
    
    def __init__(self, contracts_dir: str = "contracts"):
        self.contracts_dir = Path(contracts_dir)
        self.results = []
        self.errors = []
        
    def log(self, message: str, level: str = "INFO"):
        """Log validation messages"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.results.append(f"[{timestamp}] {level}: {message}")
        
    def error(self, message: str):
        """Log error messages"""
        self.log(message, "ERROR")
        self.errors.append(message)
        
    def success(self, message: str):
        """Log success messages"""
        self.log(message, "PASS")
        
    def load_contract(self, filepath: Path) -> Dict:
        """Load and parse YAML contract"""
        try:
            with open(filepath, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            self.error(f"Failed to load {filepath.name}: {e}")
            return None
            
    def validate_structure(self, contract: Dict, name: str) -> bool:
        """Validate basic ODCS structure"""
        required_sections = [
            'dataContractSpecification',
            'id',
            'version',
            'info',
            'schema',
            'sla',
            'quality'
        ]
        
        missing = [s for s in required_sections if s not in contract]
        if missing:
            self.error(f"{name}: Missing sections: {', '.join(missing)}")
            return False
            
        # Validate info section
        info_required = ['title', 'description', 'owner', 'contact', 'classification']
        info_missing = [f for f in info_required if f not in contract.get('info', {})]
        if info_missing:
            self.error(f"{name}: Missing info fields: {', '.join(info_missing)}")
            return False
            
        # Validate schema section
        schema = contract.get('schema', {})
        if 'properties' not in schema:
            self.error(f"{name}: Schema missing 'properties'")
            return False
            
        # Validate quality section
        quality = contract.get('quality', [])
        if not isinstance(quality, list) or len(quality) == 0:
            self.error(f"{name}: Quality rules missing or empty")
            return False
            
        self.success(f"{name}: All required ODCS sections present")
        return True
        
    def validate_scenario1_rides(self, contract: Dict) -> bool:
        """Validate Scenario 1: Ride-Share contract"""
        self.log("\\n--- Validating Scenario 1: Ride-Share ---")
        
        passed = True
        schema_props = contract.get('schema', {}).get('properties', {})
        
        # Check logical field mappings
        required_fields = [
            'ride_id', 'pickup_timestamp', 'passenger_id',
            'driver_rating', 'fare_amount', 'distance_meters'
        ]
        missing_fields = [f for f in required_fields if f not in schema_props]
        if missing_fields:
            self.error(f"Rides: Missing logical fields: {', '.join(missing_fields)}")
            passed = False
        else:
            self.success("Rides: All logical field mappings present")
            
        # Check PII tagging on passenger_id
        passenger_field = schema_props.get('passenger_id', {})
        if passenger_field.get('pii') == True:
            self.success("Rides: PII tagging on passenger_id ✓")
        else:
            self.error("Rides: passenger_id not marked as PII")
            passed = False
            
        # Check SLA freshness = 30 minutes
        sla = contract.get('sla', {})
        freshness = sla.get('freshness', {}).get('threshold', '')
        if '30 minutes' in freshness or '30 min' in freshness:
            self.success("Rides: SLA freshness = 30 minutes ✓")
        else:
            self.error(f"Rides: SLA freshness incorrect: {freshness}")
            passed = False
            
        # Check quality rules
        quality_rules = contract.get('quality', [])
        rule_names = [r.get('name', '') for r in quality_rules]
        
        required_rules = [
            ('fare_amount_non_negative', 'fare.*>=.*0'),
            ('driver_rating_range', 'driver_rating.*>=.*1.*<=.*5'),
            ('distance_not_null', 'distance')
        ]
        
        for rule_name, pattern in required_rules:
            matching_rules = [r for r in quality_rules 
                            if rule_name in r.get('name', '') or 
                            re.search(pattern, str(r.get('expression', '')), re.IGNORECASE)]
            if matching_rules:
                self.success(f"Rides: Quality rule '{rule_name}' found ✓")
            else:
                self.error(f"Rides: Quality rule '{rule_name}' missing")
                passed = False
                
        return passed
        
    def validate_scenario2_orders(self, contract: Dict) -> bool:
        """Validate Scenario 2: E-commerce contract"""
        self.log("\\n--- Validating Scenario 2: E-commerce Orders ---")
        
        passed = True
        schema_props = contract.get('schema', {}).get('properties', {})
        
        # Check order_total field with minimum 0
        order_total = schema_props.get('order_total', {})
        if order_total.get('minimum') == 0 or order_total.get('minimum') == 0.0:
            self.success("Orders: order_total has minimum: 0 ✓")
        else:
            self.error("Orders: order_total missing minimum: 0 constraint")
            passed = False
            
        # Check status enum mapping
        status_field = schema_props.get('status', {})
        status_enum = status_field.get('enum', [])
        expected_statuses = ['PAID', 'SHIPPED', 'CANCELLED']
        if set(status_enum) == set(expected_statuses):
            self.success("Orders: Status enum correctly mapped ✓")
        else:
            self.error(f"Orders: Status enum incorrect: {status_enum}")
            passed = False
            
        # Check quality rules for invalid status code rejection
        quality_rules = contract.get('quality', [])
        
        # Check for non-negative rule
        non_neg_rules = [r for r in quality_rules 
                        if 'order_total' in str(r.get('expression', '')) and 
                        '>= 0' in str(r.get('expression', ''))]
        if non_neg_rules:
            self.success("Orders: Non-negative order_total rule ✓")
        else:
            self.error("Orders: Missing non-negative order_total rule")
            passed = False
            
        # Check for status code validation (2, 5, 9)
        status_rules = [r for r in quality_rules 
                       if 'status' in r.get('name', '').lower() and
                       ('2' in str(r.get('expression', '')) or 
                        'PAID' in str(r.get('expression', '')))]
        if status_rules:
            self.success("Orders: Status code validation rule ✓")
        else:
            self.error("Orders: Missing status code validation rule")
            passed = False
            
        return passed
        
    def validate_scenario3_thermostat(self, contract: Dict) -> bool:
        """Validate Scenario 3: IoT Thermostat contract"""
        self.log("\\n--- Validating Scenario 3: IoT Thermostat ---")
        
        passed = True
        schema_props = contract.get('schema', {}).get('properties', {})
        
        # Check temperature range in schema
        temp_field = schema_props.get('temperature_c', {})
        if temp_field.get('minimum') == -30 and temp_field.get('maximum') == 60:
            self.success("Thermostat: Temperature range [-30, 60] in schema ✓")
        else:
            self.error(f"Thermostat: Temperature range incorrect in schema")
            passed = False
            
        # Check battery level range in schema
        battery_field = schema_props.get('battery_level', {})
        if battery_field.get('minimum') == 0.0 and battery_field.get('maximum') == 1.0:
            self.success("Thermostat: Battery range [0.0, 1.0] in schema ✓")
        else:
            self.error(f"Thermostat: Battery range incorrect in schema")
            passed = False
            
        # Check quality rules
        quality_rules = contract.get('quality', [])
        
        # Temperature range check
        temp_rules = [r for r in quality_rules 
                     if 'temperature' in r.get('name', '').lower() and
                     '-30' in str(r.get('expression', '')) and
                     '60' in str(r.get('expression', ''))]
        if temp_rules:
            self.success("Thermostat: Temperature range quality rule ✓")
        else:
            self.error("Thermostat: Missing temperature range quality rule")
            passed = False
            
        # Battery level check
        battery_rules = [r for r in quality_rules 
                        if 'battery' in r.get('name', '').lower() and
                        '0.0' in str(r.get('expression', '')) and
                        '1.0' in str(r.get('expression', ''))]
        if battery_rules:
            self.success("Thermostat: Battery level quality rule ✓")
        else:
            self.error("Thermostat: Missing battery level quality rule")
            passed = False
            
        return passed
        
    def validate_scenario4_fintech(self, contract: Dict) -> bool:
        """Validate Scenario 4: FinTech contract"""
        self.log("\\n--- Validating Scenario 4: FinTech Transactions ---")
        
        passed = True
        quality_rules = contract.get('quality', [])
        
        # Check for regex pattern ^[A-Z0-9]{10}$
        pattern_rules = [r for r in quality_rules 
                        if 'account' in r.get('name', '').lower() and
                        'REGEXP' in str(r.get('expression', '')) and
                        '[A-Z0-9]' in str(r.get('expression', '')) and
                        '{10}' in str(r.get('expression', ''))]
        
        if len(pattern_rules) >= 2:  # Should have rules for both source and dest
            self.success("FinTech: Regex pattern ^[A-Z0-9]{10}$ found ✓")
        else:
            self.error("FinTech: Missing or incomplete regex pattern rules")
            passed = False
            
        # Check for hard circuit breaker enforcement
        hard_enforcement = [r for r in quality_rules 
                           if r.get('enforcement') == 'hard']
        if hard_enforcement:
            self.success("FinTech: Hard circuit breaker enforcement ✓")
        else:
            self.error("FinTech: Missing hard enforcement")
            passed = False
            
        # Additional check: description mentions blocking
        blocking_rules = [r for r in quality_rules 
                         if 'BLOCK' in str(r.get('description', '')).upper()]
        if blocking_rules:
            self.success("FinTech: Hard circuit breaker documented ✓")
        else:
            self.log("FinTech: Hard circuit breaker not explicitly documented", "WARN")
            
        return passed
        
    def run_validation(self) -> bool:
        """Run all validations"""
        self.log("=" * 70)
        self.log("DATA CONTRACT VALIDATION REPORT")
        self.log(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.log("=" * 70)
        
        contracts = {
            'rides_contract.yaml': self.validate_scenario1_rides,
            'orders_contract.yaml': self.validate_scenario2_orders,
            'thermostat_contract.yaml': self.validate_scenario3_thermostat,
            'fintech_contract.yaml': self.validate_scenario4_fintech
        }
        
        all_passed = True
        
        for filename, validator_func in contracts.items():
            filepath = self.contracts_dir / filename
            
            if not filepath.exists():
                self.error(f"Contract file not found: {filename}")
                all_passed = False
                continue
                
            self.log(f"\\n{'='*70}")
            self.log(f"Validating: {filename}")
            self.log(f"{'='*70}")
            
            contract = self.load_contract(filepath)
            if contract is None:
                all_passed = False
                continue
                
            # Validate structure
            if not self.validate_structure(contract, filename):
                all_passed = False
                continue
                
            # Validate scenario-specific requirements
            if not validator_func(contract):
                all_passed = False
                
        # Summary
        self.log(f"\\n{'='*70}")
        self.log("VALIDATION SUMMARY")
        self.log(f"{'='*70}")
        
        if all_passed and len(self.errors) == 0:
            self.success("✓ ALL VALIDATIONS PASSED")
            self.log("All contracts meet assignment requirements!")
        else:
            self.error(f"✗ VALIDATION FAILED - {len(self.errors)} error(s) found")
            self.log("\\nErrors:")
            for error in self.errors:
                self.log(f"  - {error}")
                
        return all_passed
        
    def save_report(self, output_file: str = "logs/validation_report.log"):
        """Save validation report to file"""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            for line in self.results:
                f.write(line + '\\n')
                
        print(f"\\nValidation report saved to: {output_file}")


def main():
    """Main entry point"""
    validator = ContractValidator()
    
    # Run validation
    success = validator.run_validation()
    
    # Print results to console
    print("\\n".join(validator.results))
    
    # Save report
    validator.save_report()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
