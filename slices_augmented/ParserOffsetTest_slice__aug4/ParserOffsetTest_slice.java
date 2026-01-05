/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class ParserOffsetTest_slice {
    @Positive
  public void addition3(String[] a, @IndexFor("#1") int i) {
        for (int __cfwr_i43 = 0; __cfwr_i43 < 7; __cfwr_i43++) {
            return null;
        }

    @Positive
    if ((i + 5) < a.length) {
    @Positive
      @IndexFor("a") int j = i + 5;
    @Positive
    }
    @Positive
  }

    @Positive
  public void subtraction3(String[] a, @NonNegative int k) {
    @Positive
    if (k - 5 < a.length) {
    @Positive
      String s = a[k - 5];
    @Positive
      @IndexFor("a") int j = k - 5;
    @Positive
    }
    @Positive
  }

    @Positive
  public void subtraction4(String[] a, @IndexFor("#1") int i) {
    @Positive
    if (1 - i < a.length) {
      // The error on this assignment is a false positive.
      // :: error: (assignment)
    @Positive
      @IndexFor("a") int j = 1 - i;

      // :: error: (assignment)
    @Positive
      @LTLengthOf(value = "a", offset = "1") int k = i;
    @Positive
    }
    @Positive
  }

    @Positive
  public void subtraction5(String[] a, int i) {
    @Positive
    if (1 - i < a.length) {
      // :: error: (assignment)
    @Positive
      @IndexFor("a") int j = i;
    @Positive
    }
    @Positive
  }

    public Long __cfwr_temp315(byte __cfwr_p0, int __cfwr_p1, Integer __cfwr_p2) {
        while (true) {
            if (false || (null % null)) {
            for (int __cfwr_i76 = 0; __cfwr_i76 < 1; __cfwr_i76++) {
            for (int __cfwr_i20 = 0; __cfwr_i20 < 5; __cfwr_i20++) {
            for (int __cfwr_i45 = 0; __cfwr_i45 < 7; __cfwr_i45++) {
            if (false && false) {
            for (int __cfwr_i74 = 0; __cfwr_i74 < 6; __cfwr_i74++) {
            boolean __cfwr_item46 = false;
        }
        }
        }
        }
        }
        }
            break; // Prevent infinite loops
        }
        try {
            while (true) {
            while ((null % 26.20f)) {
            try {
            if (true && true) {
            try {
            Character __cfwr_obj10 = null;
        } catch (Exception __cfwr_e36) {
            // ignore
        }
        }
        } catch (Exception __cfwr_e99) {
            // ignore
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e27) {
            // ignore
        }
        while (true) {
            if (false || true) {
            try {
            for (int __cfwr_i17 = 0; __cfwr_i17 < 9; __cfwr_i17++) {
            return null;
        }
        } catch (Exception __cfwr_e45) {
            // ignore
        }
        }
            break; // Prevent infinite loops
        }
        return null;
    }
    private byte __cfwr_compute352(Boolean __cfwr_p0, Float __cfwr_p1) {
        return null;
        return null;
    }
}