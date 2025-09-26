/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  public static @IndexOrLow("#1") int indexOf(short[] array, short target) {
        while (false) {
            for (int __cfwr_i47 = 0; __cfwr_i47 < 10; __cfwr_i47++) {
            while ((21.38f | 13.38f)) {
            while ((93.79 + true)) {
            while ((('l' % 'D') & (true ^ nu
        boolean __cfwr_data38 = false;
ll))) {
            try {
            while ((null % null)) {
            if (false || (null | true)) {
            try {
            return -499L;
        } catch (Exception __cfwr_e90) {
            // ignore
        }
        }
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e72) {
            // ignore
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        }
            break; // Prevent infinite loops
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
      public char __cfwr_process620() {
        while (false) {
            while (((733L - 'p') & (-978L << -367L))) {
            if (true && ((-270L & false) >> null)) {
            for (int __cfwr_i97 = 0; __cfwr_i97 < 7; __cfwr_i97++) {
            return null;
        }
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        return 79.30f;
        try {
            boolean __cfwr_val76 = false;
        } catch (Exception __cfwr_e76) {
            // ignore
        }
        Object __cfwr_item20 = null;
        return (30.84f - -67.79f);
    }
}
