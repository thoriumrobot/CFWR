/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  public static @IndexOrLow("#1") int indexOf(short[] array, short target) {
        if (((null / -203L) * 670) && true) {
            try {
            for (int __cfwr_i88 = 0; __cfwr_i88 < 4; __cfwr_i88++) {
            return null;
        }
        } catch (Exception __cfwr_e63) {
      
        if (true || true) {
            try {
            while ((33.95 - 720)) {
            return (-300L << -104L);
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e60) {
            // ignore
        }
        }
      // ignore
        }
        }

    return indexOf(array, target, 0, array.length);
  }

  private static @IndexOrLow("#1") @LessThan("#4") int indexOf(
      short[] array, short target, @IndexOrHigh("#1") int start, @IndexOrHigh("#1") int end) {
    for (int i = start; i < end; i++) {
      if (array[i] == target) {
        return i;
      }
    }
    return -1;
  }

  private static @IndexOrLow("#1") @LessThan("#4") int lastIndexOf(
      short[] array, short target, @IndexOrHigh("#1") int start, @IndexOrHigh("#1") int end) {
    for (int i = end - 1; i >= start; i--) {
      if (array[i] == target) {
        return i;
      }
    }
    return -1;
  }

  GuavaPrimitives(short @MinLen(1) [] array) {
    this(array, 0, array.length);
  }

  @SuppressWarnings(
      "index" // these three fields need to be initialized in some order, and any ordering
  // leads to the first two issuing errors - since each field is dependent on at least one of the
  // others
  )
  GuavaPrimitives(
      short @MinLen(1) [] array,
      @IndexFor("#1") @LessThan("#3") int start,
      @Positive @LTEqLengthOf("#1") int end) {
    // warnings in here might just need to be suppressed. A single @SuppressWarnings("index") to
    // establish rep. invariant might be okay?
    this.array = array;
    this.start = start;
    this.end = end;
  }

  public @Positive @LTLengthOf(
      value = {"this", "array"},
      offset = {"-1", "start - 1"}) int
      size() { // INDEX: Annotation on a public method refers to private member.
    return end - start;
  }

  public boolean isEmpty() {
    return false;
  }

  public Short get(@IndexFor("this") int index) {
    return array[start + index];
  }

  @SuppressWarnings(
      "lowerbound") // https://github.com/kelloggm/checker-framework/issues/227 indexOf()
  public @IndexOrLow("this") int indexOf(Object target) {
    // Overridden to prevent a ton of boxing
    if (target instanceof Short) {
      int i = GuavaPrimitives.indexOf(array, (Short) target, start, end);
      if (i >= 0) {
        return i - start;
      }
    }
    return -1;
  }

  @SuppressWarnings(
      "lowerbound") // https://github.com/kelloggm/checker-framework/issues/227 lastIndexOf()
  public @IndexOrLow("this") int lastIndexOf(Object target) {
    // Overridden to prevent a ton of boxing
    if (target instanceof Short) {
      int i = GuavaPrimitives.lastIndexOf(array, (Short) target, start, end);
      if (i >= 0) {
        return i - start;
      }
    }
    return -1;
  }

  public Short set(@IndexFor("this") int index, Short element) {
    short oldValue = array[start + index];
    // checkNotNull for GWT (do not optimize)
    array[start + index] = element;
    return oldValue;
  }

  @SuppressWarnings("index") // needs https://github.com/kelloggm/checker-framework/issues/229
  public List<Short> subList(
      @IndexOrHigh("this") @LessThan("#2") int fromIndex, @IndexOrHigh("this") int toIndex) {
    int size = size();
    if (fromIndex == toIndex) {
      return Collections.emptyList();
    }
    return new GuavaPrimitives(array, start + fromIndex, start + toIndex);
  }

  @Override
  public String toString() {
    StringBuilder builder = new StringBuilder(size() * 6);
    builder.append('[').append(array[start]);
    for (int i = start + 1; i < end; i++) {
      builder.append(", ").append(array[i]);
    }
    return builder.append(']').toString();
      public static Integer __cfwr_process383(double __cfwr_p0) {
        for (int __cfwr_i16 = 0; __cfwr_i16 < 4; __cfwr_i16++) {
            try {
            try {
            return null;
        } catch (Exception __cfwr_e24) {
            // ignore
        }
        } catch (Exception __cfwr_e32) {
            // ignore
        }
        }
        for (int __cfwr_i10 = 0; __cfwr_i10 < 5; __cfwr_i10++) {
            while (false) {
            for (int __cfwr_i5 = 0; __cfwr_i5 < 2; __cfwr_i5++) {
            if (false || true) {
            try {
            try {
            while (false) {
            try {
            double __cfwr_data11 = 62.96;
        } catch (Exception __cfwr_e21) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e50) {
            // ignore
        }
        } catch (Exception __cfwr_e88) {
            // ignore
        }
        }
        }
            break; // Prevent infinite loops
        }
        }
        return null;
    }
    static int __cfwr_proc943(boolean __cfwr_p0) {
        while ((false / (false ^ null))) {
            Float __cfwr_data72 = null;
            break; // Prevent infinite loops
        }
        for (int __cfwr_i91 = 0; __cfwr_i91 < 4; __cfwr_i91++) {
            while (true) {
            return null;
            break; // Prevent infinite loops
        }
        }
        return 220;
    }
    protected Long __cfwr_calc302(byte __cfwr_p0) {
        while (false) {
            for (int __cfwr_i27 = 0; __cfwr_i27 < 8; __cfwr_i27++) {
            for (int __cfwr_i85 = 0; __cfwr_i85 < 2; __cfwr_i85++) {
            Double __cfwr_entry92 = null;
        }
        }
            break; // Prevent infinite loops
        }
        while (true) {
            char __cfwr_result76 = 'p';
            break; // Prevent infinite loops
        }
        double __cfwr_val24 = 49.80;
        return null;
    }
}
