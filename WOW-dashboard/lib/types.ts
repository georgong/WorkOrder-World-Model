/**
 * Shared TypeScript types for the dashboard.
 */

export interface ScheduleMetrics {
  overall_risk_score: number;
  workload_imbalance_score: number;
  total_assignments: number;
  avg_predicted_hours: number;
  median_predicted_hours: number;
  most_overloaded_engineer: string;
  highest_risk_district: string;
  highest_risk_department: string;
}

export interface AssignmentPrediction {
  assignment_id: string;
  pred_completion_hours: number;
  risk_score: number;
  top_factors: string[];
}

export interface ChartData {
  risk_histogram: Record<string, any>[];
  risk_by_district: Record<string, any>[];
  workload_by_engineer: Record<string, any>[];
  risk_by_department: Record<string, any>[];
}

export interface PredictResponse {
  schedule_metrics: ScheduleMetrics;
  assignment_predictions: AssignmentPrediction[];
  charts: ChartData;
  metadata: Record<string, any>;
}

export interface GraphResponse {
  nodes: any[];
  edges: any[];
  node_types: string[];
  edge_types: string[];
}
