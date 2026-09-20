CC      := gcc
CFLAGS  := -Wall -Wextra -std=c11 -O2 -g
LDFLAGS := -lm

SRC_DIR   := src
TEST_DIR  := tests
BUILD_DIR := build

SRCS := $(wildcard $(SRC_DIR)/*.c)
OBJS := $(patsubst $(SRC_DIR)/%.c,$(BUILD_DIR)/%.o,$(SRCS))

TARGET := mlp

.PHONY: all clean test debug

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# --- TESTS ---
TEST_SRCS := $(wildcard $(TEST_DIR)/*.c)
TEST_OBJS := $(patsubst $(TEST_DIR)/%.c,$(BUILD_DIR)/%.o,$(TEST_SRCS))
CORE_SRCS := $(filter-out $(SRC_DIR)/main.c,$(SRCS))
CORE_OBJS := $(patsubst $(SRC_DIR)/%.c,$(BUILD_DIR)/%.o,$(CORE_SRCS))

$(BUILD_DIR)/%.o: $(TEST_DIR)/%.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -I$(SRC_DIR) -c $< -o $@

test: $(CORE_OBJS) $(TEST_OBJS)
	@for test_obj in $(TEST_OBJS); do \
		test_exe=$${test_obj%.o}; \
		echo "-> Compilation du test: $$test_exe"; \
		$(CC) $(CFLAGS) -o $$test_exe $$test_obj $(CORE_OBJS) $(LDFLAGS); \
		echo " >>> Exécution de $$test_exe"; \
		$$test_exe; \
		echo ""; \
	done

debug: CFLAGS += -fsanitize=address,undefined -DDEBUG
debug: LDFLAGS += -fsanitize=address,undefined
debug: clean all

clean:
	rm -rf $(BUILD_DIR) $(TARGET) $(TEST_DIR)/*_runner
