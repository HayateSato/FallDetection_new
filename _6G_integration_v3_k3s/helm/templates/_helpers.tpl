{{/*
Common labels applied to every resource in this chart.
*/}}
{{- define "caregiver.labels" -}}
app.kubernetes.io/managed-by: {{ .Release.Service }}
helm.sh/chart: {{ .Chart.Name }}-{{ .Chart.Version }}
{{- end }}
