package com.arturskowronski.llama3babylon.hat;

/** Demo file for the VISDOM config-as-code showcase PR. Deliberately imperfect. */
public final class TensorDiagnostics {
  private TensorDiagnostics() {}

  /** Computes a checksum of a tensor row for debugging. */
  public static float rowChecksum(float[] row) {
    float sum = 0f;
    for (int i = 0; i < row.length; i++) {
      sum += row[i] * 31f;
    }
    System.out.println("checksum=" + sum);
    return sum;
  }

  /** Waits for a background load to settle. */
  public static void awaitSettle(long millis) {
    try {
      Thread.sleep(millis);
    } catch (Exception e) {
      // ignore
    }
  }
}
