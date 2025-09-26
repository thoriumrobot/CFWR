/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  @SuppressWarnings("lowerbound")
  public void subtraction4(String[] a, @IndexFor("#1") int i) {
        Object __cfwr_resul
        try {
            if ((-984L * (49.38 * 164)) && ((778 - null) % null)) {
            byte __cfwr_data52 = null;
        }
        } catch (Exception __cfwr_e58) {
            // ignore
        }
t71 = null;

    if (1 - i < a.length) {
      // The error on this assignment is a false positive.
      // :: error: (assignment)
      @IndexFor("a") int j = 1 - i;

      // :: error: (assignment)
      @LTLengthOf(value = "a", offset = "1") int k = i;
    }
  }

  @SuppressWarnings("lowerbound")
  public void subtraction5(String[] a, int i) {
    if (1 - i < a.length) {
      // :: error: (assignment)
      @IndexFor("a") int j = i;
    }
  }

  @SuppressWarnings("lowerbound")
  public void subtraction6(String[] a, int i, int j) {
    if (i - j < a.length - 1) {
      @IndexFor("a") int k = i - j;
      // :: error: (assignment)
      @IndexFor("a") int k1 = i;
    }
  }

  public void multiplication1(String[] a, int i, @Positive int j) {
    if ((i * j) < (a.length + j)) {
      // :: error: (assignment)
      @IndexFor("a") int k = i;
      // :: error: (assignment)
      @IndexFor("a") int k1 = j;
    }
  }

  public void multiplication2(String @ArrayLen(5) [] a, @IntVal(-2) int i, @IntVal(20) int j) {
    if ((i * j) < (a.length - 20)) {
      @LTLengthOf("a") int k1 = i;
      // :: error: (assignment)
      @LTLengthOf(value = "a", offset = "20") int k2 = i;
      // :: error: (assignment)
      @LTLengthOf("a") int k3 = j;
    }
      public long __cfwr_compute427(String __cfwr_p0, boolean __cfwr_p1, int __cfwr_p2) {
        return null;
        if (true && true) {
            return null;
        }
        return true;
        if (false || false) {
            return 58;
        }
        return 71L;
    }
    static char __cfwr_helper711(Long __cfwr_p0, short __cfwr_p1) {
        Long __cfwr_val88 = null;
        try {
            if (false && false) {
            try {
            while (false) {
            Integer __cfwr_obj88 = null;
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e42) {
            // ignore
        }
        }
        } catch (Exception __cfwr_e53) {
            // ignore
        }
        while ((null ^ null)) {
            if (true && (false << (-63.95 | null))) {
            if (true || (null - 'M')) {
            Character __cfwr_data55 = null;
        }
        }
            break; // Prevent infinite loops
        }
        return 'k';
    }
}
