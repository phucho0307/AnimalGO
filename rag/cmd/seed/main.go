// Command seed builds the RAG knowledge base: it asks Gemini for batches of
// US species (as strict JSON) and inserts each as a "document" row in
// species_docs. Embeddings are added later by cmd/embed. Run from the rag/ dir:
//
//	go run ./cmd/seed
package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/jackc/pgx/v5"
)

// Species is one "document" in our corpus. The `description` is what we embed;
// scientific_name/category are metadata we key and filter on.
type Species struct {
	CommonName     string `json:"common_name"`
	ScientificName string `json:"scientific_name"`
	Category       string `json:"category"`
	Description    string `json:"description"`
}

// How many of each category to generate. ~110 species total for a first corpus.
var categories = []struct {
	Name  string
	Count int
}{
	{"bird", 40},
	{"mammal", 30},
	{"reptile", 15},
	{"amphibian", 10},
	{"insect", 15},
}

const geminiModel = "gemini-2.5-flash"

func main() {
	loadDotEnv("../backend/.env") // reuse the existing project secrets

	dbURL := os.Getenv("DATABASE_URL")
	geminiKey := os.Getenv("GEMINI_API_KEY")
	if dbURL == "" || geminiKey == "" {
		log.Fatal("need DATABASE_URL and GEMINI_API_KEY (from backend/.env)")
	}
	// pgx speaks plain postgres URLs; drop SQLAlchemy's "+asyncpg" driver tag.
	dbURL = strings.Replace(dbURL, "postgresql+asyncpg://", "postgresql://", 1)

	ctx := context.Background()
	conn, err := pgx.Connect(ctx, dbURL)
	if err != nil {
		log.Fatalf("db connect: %v", err)
	}
	defer conn.Close(ctx)

	total := 0
	for _, c := range categories {
		fmt.Printf("- asking Gemini for %d %s species...\n", c.Count, c.Name)
		list, err := generateSpecies(geminiKey, c.Name, c.Count)
		if err != nil {
			log.Printf("  gemini(%s) failed: %v (skipping)", c.Name, err)
			continue
		}
		inserted := 0
		for _, s := range list {
			if strings.TrimSpace(s.ScientificName) == "" || strings.TrimSpace(s.Description) == "" {
				continue
			}
			// ON CONFLICT keeps re-runs idempotent — a species already in the
			// corpus isn't duplicated.
			tag, err := conn.Exec(ctx,
				`INSERT INTO species_docs (common_name, scientific_name, category, description)
				 VALUES ($1,$2,$3,$4)
				 ON CONFLICT (scientific_name) DO NOTHING`,
				s.CommonName, s.ScientificName, c.Name, s.Description)
			if err != nil {
				log.Printf("  insert %q failed: %v", s.ScientificName, err)
				continue
			}
			inserted += int(tag.RowsAffected())
		}
		fmt.Printf("  inserted %d new %s (of %d returned)\n", inserted, c.Name, len(list))
		total += inserted
	}

	var count int
	_ = conn.QueryRow(ctx, "select count(*) from species_docs").Scan(&count)
	fmt.Printf("\nDone. Added %d new; species_docs now holds %d rows.\n", total, count)
}

// generateSpecies asks Gemini for n species of a category, forced into a strict
// JSON array via responseSchema (no prose, no markdown fences to clean up).
func generateSpecies(apiKey, category string, n int) ([]Species, error) {
	prompt := fmt.Sprintf(
		"List %d common wild %s species found in the United States. For each, give "+
			"its common_name, scientific_name (Latin binomial), category (%q), and a "+
			"2-3 sentence description covering appearance (size, colors, distinctive "+
			"markings), habitat, and behavior - the kind of text someone would use to "+
			"recognize or search for the animal.", n, category, category)

	reqBody := map[string]any{
		"contents": []any{map[string]any{
			"parts": []any{map[string]any{"text": prompt}},
		}},
		"generationConfig": map[string]any{
			"responseMimeType": "application/json",
			"responseSchema": map[string]any{
				"type": "ARRAY",
				"items": map[string]any{
					"type": "OBJECT",
					"properties": map[string]any{
						"common_name":     map[string]any{"type": "STRING"},
						"scientific_name": map[string]any{"type": "STRING"},
						"category":        map[string]any{"type": "STRING"},
						"description":     map[string]any{"type": "STRING"},
					},
					"required": []string{"common_name", "scientific_name", "description"},
				},
			},
		},
	}
	buf, _ := json.Marshal(reqBody)

	url := fmt.Sprintf("https://generativelanguage.googleapis.com/v1beta/models/%s:generateContent", geminiModel)
	req, _ := http.NewRequest("POST", url, bytes.NewReader(buf))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("x-goog-api-key", apiKey)

	resp, err := (&http.Client{Timeout: 90 * time.Second}).Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != 200 {
		return nil, fmt.Errorf("gemini http %d: %s", resp.StatusCode, snippet(raw))
	}

	// The JSON array we asked for is nested inside the Gemini envelope.
	var env struct {
		Candidates []struct {
			Content struct {
				Parts []struct {
					Text string `json:"text"`
				} `json:"parts"`
			} `json:"content"`
		} `json:"candidates"`
	}
	if err := json.Unmarshal(raw, &env); err != nil {
		return nil, fmt.Errorf("decode envelope: %w", err)
	}
	if len(env.Candidates) == 0 || len(env.Candidates[0].Content.Parts) == 0 {
		return nil, fmt.Errorf("empty gemini response")
	}
	var species []Species
	if err := json.Unmarshal([]byte(env.Candidates[0].Content.Parts[0].Text), &species); err != nil {
		return nil, fmt.Errorf("decode species json: %w", err)
	}
	return species, nil
}

func snippet(b []byte) string {
	if len(b) > 300 {
		return string(b[:300])
	}
	return string(b)
}

// loadDotEnv reads simple KEY=VALUE lines into the process env (only if unset).
func loadDotEnv(path string) {
	data, err := os.ReadFile(path)
	if err != nil {
		return
	}
	for _, line := range strings.Split(string(data), "\n") {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		k, v, ok := strings.Cut(line, "=")
		if !ok {
			continue
		}
		k, v = strings.TrimSpace(k), strings.Trim(strings.TrimSpace(v), `"'`)
		if os.Getenv(k) == "" {
			os.Setenv(k, v)
		}
	}
}
