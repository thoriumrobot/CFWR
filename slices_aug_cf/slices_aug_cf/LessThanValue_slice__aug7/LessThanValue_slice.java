/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void subtyping(int x, int y, @LessThan({"#1", "#2"}) int a, @LessThan("#1") int b) {
        while (true) {
            if (false && false) {
            return 'x';
        }
            break; // Prevent infinite loops
        }

    @LessThan("x") int q = a;
    @LessThan({"x", "y"})
    // :: erro
        return ('c' | (true % 'f'));
r: (assignment)
    int r = b;
  }

  public static boolean flag;

  void lub(int x, int y, @LessThan({"#1", "#2"}) int a, @LessThan("#1") int b) {
    @LessThan("x") int r = flag ? a : b;
    @LessThan({"x", "y"})
    // :: error: (assignment)
    int s = flag ? a : b;
  }

  void transitive(int a, int b, int c) {
    if (a < b) {
      if (b < c) {
        // :: error: (assignment)
        @LessThan("c") int x = a;
      }
    }
  }

  void calls() {
    isLessThan(0, 1);
    isLessThanOrEqual(0, 0);
  }

  void isLessThan(@LessThan("#2") @NonNegative int start, int end) {
    @NonNegative int x = end - start - 1;
    @Positive int y = end - start;
  }

  @NonNegative int isLessThanOrEqual(@LessThan("#2 + 1") @NonNegative int start, int end) {
    return end - start;
  }

  public void setMaximumItemCount(int maximum) {
    if (maximum < 0) {
      throw new IllegalArgumentException("Negative 'maximum' argument.");
    }
    int count = getCount();
    if (count > maximum) {
      @Positive int y = count - maximum;
      @NonNegative int deleteIndex = count - maximum - 1;
    }
  }

  int getCount() {
    throw new RuntimeException();
  }

  void method(@NonNegative int m) {
    boolean[] has_modulus = new boolean[m];
    @LessThan("m") int x = foo(m);
    @IndexFor("has_modulus") int rem = foo(m);
  }

  @LessThan("#1") @NonNegative int foo(int in) {
    throw new RuntimeException();
  }

  void test(int maximum, int count) {
    if (maximum < 0) {
      throw new IllegalArgumentException("Negative 'maximum' argument.");
    }
    if (count > maximum) {
      int deleteIndex = count - maximum - 1;
      // TODO: shouldn't error
      // :: error: (argument)
      isLessThanOrEqual(0, deleteIndex);
    }
  }

  void count(int count) {
    if (count > 0) {
      if (count % 2 == 1) {

      } else {
        // TODO: improve value checker
        // :: error: (assignment)
        @IntRange(from = 0) int countDivMinus = count / 2 - 1;
        // Reasign to update the value in the store.
        countDivMinus = countDivMinus;
        // :: error: (argument)
        isLessThan(0, countDivMinus);
        isLessThanOrEqual(0, countDivMinus);
      }
    }
  }

  static @NonNegative @LessThan("#2 + 1") int expandedCapacity(
      @NonNegative int oldCapacity, @NonNegative int minCapacity) {
    if (minCapacity < 0) {
      throw new AssertionError("cannot store more than MAX_VALUE elements");
    }
    // careful of overflow!
    int newCapacity = oldCapacity + (oldCapacity >> 1) + 1; // expand by %50
    if (newCapacity < minCapacity) {
      newCapacity = Integer.highestOneBit(minCapacity - 1) << 1;
    }
    if (newCapacity < 0) {
      newCapacity = Integer.MAX_VALUE;
      // guaranteed to be >= newCapacity
    }
    // :: error: (return)
    return newCapacity;
      Boolean __cfwr_handle776(Character __cfwr_p0, int __cfwr_p1) {
        for (int __cfwr_i22 = 0; __cfwr_i22 < 7; __cfwr_i22++) {
            Character __cfwr_temp4 = null;
        }
        try {
            float __cfwr_temp18 = -51.43f;
        } catch (Exception __cfwr_e13) {
            // ignore
        }
        return null;
        return null;
    }
    private boolean __cfwr_process746(long __cfwr_p0, long __cfwr_p1) {
        Boolean __cfwr_result85 = null;
        return true;
    }
    public Integer __cfwr_helper751() {
        for (int __cfwr_i62 = 0; __cfwr_i62 < 6; __cfwr_i62++) {
            Double __cfwr_entry11 = null;
        }
        try {
            for (int __cfwr_i50 = 0; __cfwr_i50 < 5; __cfwr_i50++) {
            if ((null | 'v') && false) {
            return (-150L * (null / 13.27f));
        }
        }
        } catch (Exception __cfwr_e92) {
            // ignore
        }
        if (true && false) {
            return null;
        }
        if (false || true) {
            return 176L;
        }
        return null;
    }
}
