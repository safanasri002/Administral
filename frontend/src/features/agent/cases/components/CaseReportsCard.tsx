import { Check, Minus } from 'lucide-react';

import { SectionHeader } from '@/components/shared';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Badge } from '@/components/ui/badge';
import { Card, CardContent, CardHeader } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import { cn } from '@/lib/utils';
import { formatDate } from '@/lib/utils';
import type { CoherenceReport, CompletenessReport } from '@/types';
import type { SubmissionVerdict } from '@/services/inconsistencyService';
import {
  ANOMALY_TONE,
  REPORT_OUTCOME_LABEL,
  REPORT_OUTCOME_TONE,
} from '@/features/agent/lib/casePresentation';

/**
 * The two pipeline verdicts, rendered verbatim.
 *
 * Both reports are produced upstream of the Agent Portal. `completionRate`,
 * `outcome` and every anomaly arrive computed — this file re-derives none of
 * them, it only lays them out.
 */

/**
 * Drops to `h3` when the card sits inside a titled group, so the outline stays
 * nested rather than flattening two levels onto `h2`.
 */
type ReportHeadingLevel = 'h2' | 'h3';

export interface CompletenessReportCardProps {
  report?: CompletenessReport;
  headingLevel?: ReportHeadingLevel;
}

export function CompletenessReportCard({
  report,
  headingLevel = 'h2',
}: CompletenessReportCardProps) {
  return (
    <Card>
      <CardHeader>
        <SectionHeader
          title="Contrôle de complétude"
          as={headingLevel}
          action={
            report && (
              <Badge tone={REPORT_OUTCOME_TONE[report.outcome]}>
                {REPORT_OUTCOME_LABEL[report.outcome]}
              </Badge>
            )
          }
        />
      </CardHeader>
      <CardContent>
        {!report ? (
          <p className="text-body-sm text-on-surface-variant">
            Le contrôle de complétude n’a pas encore été effectué pour ce dossier.
          </p>
        ) : (
          <>
            <div className="mb-4">
              <div className="mb-1 flex items-baseline justify-between">
                <span className="text-body-sm text-on-surface-variant">Pièces reçues</span>
                <span className="text-label-md text-on-surface">{report.completionRate} %</span>
              </div>
              <Progress value={report.completionRate} />
            </div>

            <ul className="space-y-1">
              {report.items.map((item) => (
                <li key={item.id} className="flex items-center gap-2 py-1">
                  <span
                    className={cn(
                      'flex size-5 shrink-0 items-center justify-center rounded-full',
                      item.received
                        ? 'bg-success-surface text-success'
                        : 'bg-surface-highest text-outline',
                    )}
                  >
                    {item.received ? (
                      <Check className="size-3" aria-hidden="true" />
                    ) : (
                      <Minus className="size-3" aria-hidden="true" />
                    )}
                  </span>
                  <span
                    className={cn(
                      'text-body-sm',
                      item.received ? 'text-on-surface' : 'text-on-surface-variant',
                    )}
                  >
                    {item.label}
                    <span className="sr-only">{item.received ? ' — reçue' : ' — manquante'}</span>
                  </span>
                  {item.required && !item.received && (
                    <Badge tone="warning" className="ml-auto">
                      Obligatoire
                    </Badge>
                  )}
                </li>
              ))}
            </ul>

            <p className="mt-4 text-body-sm text-on-surface-variant">
              Contrôlé le {formatDate(report.checkedAt)}.
            </p>
          </>
        )}
      </CardContent>
    </Card>
  );
}

export interface CoherenceReportCardProps {
  report?: CoherenceReport;
  headingLevel?: ReportHeadingLevel;
}

const VERDICT_TONE: Record<SubmissionVerdict['statut'], 'success' | 'warning' | 'neutral'> = {
  coherent: 'success',
  ecarts_non_resolus: 'warning',
  non_verifiable: 'neutral',
};

interface InconsistencyVerdictCardProps {
  verdict: SubmissionVerdict | null | undefined;
  headingLevel?: 'h2' | 'h3';
}

/**
 * Cohérence des pièces telles que soumises : un verdict figé à l'envoi, sans
 * détail. Le citoyen a pu vérifier et corriger son dossier avant l'envoi ; ces
 * vérifications lui appartiennent et ne remontent pas ici — seul l'état final
 * est communiqué (minimisation des données).
 */
export function InconsistencyVerdictCard({
  verdict,
  headingLevel = 'h2',
}: InconsistencyVerdictCardProps) {
  const label = !verdict
    ? null
    : verdict.statut === 'coherent'
      ? 'Pièces cohérentes'
      : verdict.statut === 'non_verifiable'
        ? 'Non vérifiable'
        : `${verdict.nbEcarts} écart${verdict.nbEcarts > 1 ? 's' : ''} non résolu${verdict.nbEcarts > 1 ? 's' : ''}`;
  return (
    <Card>
      <CardHeader>
        <SectionHeader
          title="Cohérence des pièces soumises"
          as={headingLevel}
          action={verdict && label && <Badge tone={VERDICT_TONE[verdict.statut]}>{label}</Badge>}
        />
      </CardHeader>
      <CardContent className="space-y-2 text-body-sm text-on-surface-variant">
        {!verdict ? (
          <p>Aucun verdict figé pour ce dossier (soumis avant la mise en place du contrôle).</p>
        ) : (
          <>
            {verdict.statut === 'ecarts_non_resolus' && (
              <p>
                Gravité élevée : {verdict.nbElevee} · moyenne : {verdict.nbMoyenne} · faible :{' '}
                {verdict.nbFaible}
              </p>
            )}
            <p>
              {verdict.nbDocuments} pièce{verdict.nbDocuments > 1 ? 's' : ''} analysée
              {verdict.nbDocuments > 1 ? 's' : ''} par règles déterministes, le{' '}
              {formatDate(verdict.calculeLe)}.
            </p>
            <p>
              Le détail des écarts reste au citoyen, qui a pu les vérifier avant l’envoi : seul
              l’état du dossier soumis est communiqué.
            </p>
          </>
        )}
      </CardContent>
    </Card>
  );
}

export function CoherenceReportCard({ report, headingLevel = 'h2' }: CoherenceReportCardProps) {
  return (
    <Card>
      <CardHeader>
        <SectionHeader
          title="Contrôle de cohérence"
          as={headingLevel}
          action={
            report && (
              <Badge tone={REPORT_OUTCOME_TONE[report.outcome]}>
                {REPORT_OUTCOME_LABEL[report.outcome]}
              </Badge>
            )
          }
        />
      </CardHeader>
      <CardContent>
        {!report ? (
          <p className="text-body-sm text-on-surface-variant">
            Le contrôle de cohérence n’a pas encore été effectué pour ce dossier.
          </p>
        ) : (
          <>
            {typeof report.coherenceScore === 'number' && (
              <div className="mb-4">
                <div className="mb-1 flex items-baseline justify-between">
                  <span className="text-body-sm text-on-surface-variant">
                    Score de cohérence global
                  </span>
                  <span className="text-label-md text-on-surface">{report.coherenceScore} / 100</span>
                </div>
                <Progress value={report.coherenceScore} />
              </div>
            )}

            {report.aiExplanation && (
              <p className="mb-4 rounded-lg bg-surface-low p-3 text-body-sm text-on-surface-variant">
                {report.aiExplanation}
              </p>
            )}

            {report.anomalies.length === 0 ? (
              <p className="text-body-sm text-on-surface-variant">
                Aucune incohérence détectée entre les données déclarées et les pièces fournies.
              </p>
            ) : (
              <ul className="space-y-3">
                {report.anomalies.map((anomaly) => (
                  <li key={anomaly.id}>
                    <Alert tone={ANOMALY_TONE[anomaly.severity]}>
                      <div>
                        <AlertTitle>{anomaly.field}</AlertTitle>
                        <AlertDescription>
                          {anomaly.message}
                          <span className="mt-1 block text-body-sm text-on-surface-variant">
                            Déclaré : {anomaly.declaredValue} · Constaté : {anomaly.observedValue}
                          </span>
                        </AlertDescription>
                      </div>
                    </Alert>
                  </li>
                ))}
              </ul>
            )}
          </>
        )}
      </CardContent>
    </Card>
  );
}
