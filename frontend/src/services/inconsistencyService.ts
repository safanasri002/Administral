import { apiClient } from '@/services/apiClient';

/** Severity of a discrepancy, as the backend rules grade it. */
export type InconsistencySeverity = 'elevee' | 'moyenne' | 'faible';

/** What the local agent thought of an ambiguous discrepancy. */
export type AgentOpinion = 'confirme' | 'infirme' | 'incertain';

export interface InconsistencyDocumentValue {
  fichier: string;
  valeur: string | null;
  /** Verbatim line of the document the value was read from. */
  preuve: string | null;
}

export interface Inconsistency {
  id: string;
  /** Declared profile vs one document, or two documents against each other. */
  portee: 'profil_document' | 'document_document';
  /** Information family: `statut_occupation`, `adresse`, `identite`, `loyer`, `surface`. */
  champ: string;
  gravite: InconsistencySeverity;
  valeurProfil: string | null;
  valeursDocuments: InconsistencyDocumentValue[];
  message: string;
  /** Always `a_verifier`: the engine flags, the citizen decides. */
  statut: 'a_verifier';
  source: 'regle' | 'metadonnees' | 'agent_ia';
  avisAgent: AgentOpinion | null;
  explicationAgent: string | null;
}

export interface InconsistencyAgentStatus {
  active: boolean;
  disponible: boolean;
  modele: string | null;
  utilise: boolean;
  note: string | null;
}

export interface InconsistencyReport {
  dossierId: string;
  genereLe: string;
  nbDocumentsAnalyses: number;
  incoherences: Inconsistency[];
  /** What could not be checked and why — absence of a finding is not proof. */
  pointsNonVerifiables: string[];
  agentIa: InconsistencyAgentStatus;
  rapportTexte: string;
}

/**
 * What the administration sees of the documents' coherence: a verdict frozen at
 * submission, status and counts only — never the discrepancies themselves, nor
 * the checks the citizen ran before sending.
 */
export interface SubmissionVerdict {
  statut: 'coherent' | 'ecarts_non_resolus' | 'non_verifiable';
  nbEcarts: number;
  nbElevee: number;
  nbMoyenne: number;
  nbFaible: number;
  nbDocuments: number;
  nbPointsNonVerifiables: number;
  methode: 'regles';
  version: string;
  calculeLe: string;
}

/**
 * Inconsistencies between the signed-in citizen's profile and their documents.
 *
 * Runs entirely on the server (`app/modules/ai/inconsistency`): metadata, text
 * and OCR are read locally, and ambiguous cases go to a Mistral model served
 * locally. Expect up to a couple of minutes when the local model is consulted —
 * it runs on CPU.
 */
export const inconsistencyService = {
  getReport: () => apiClient.get<InconsistencyReport>('/citizen/incoherences'),
};
