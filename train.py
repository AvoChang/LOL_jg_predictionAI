def train_model():
    """모델 학습 실행"""
    try:
        # 데이터 로드
        print("데이터 로드 중...")
        X_train = np.load('data/processed/X_train.npy')
        y_train = np.load('data/processed/y_train.npy')
        X_val = np.load('data/processed/X_val.npy')
        y_val = np.load('data/processed/y_val.npy')
        
        # 데이터 형태 확인
        print(f"학습 데이터 형태: {X_train.shape}")
        print(f"검증 데이터 형태: {X_val.shape}")
        
        # 모델 생성
        print("모델 생성 중...")
        model = Sequential([
            # 입력 레이어
            Input(shape=(X_train.shape[1],)),
            
            # 첫 번째 Dense 블록
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            
            # 두 번째 Dense 블록
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            
            # 세 번째 Dense 블록
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            
            # 출력 레이어
            Dense(1, activation='sigmoid')
        ])
        
        # 모델 컴파일
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', AUC()]
        )
        
        # 콜백 설정
        callbacks = [
            ModelCheckpoint(
                'models/best_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=0.00001,
                verbose=1
            )
        ]
        
        # 모델 학습
        print("모델 학습 시작...")
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=100,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )
        
        # 학습 결과 저장
        print("학습 결과 저장 중...")
        save_training_history(history)
        
        # 최종 모델 저장
        model.save('models/final_model.h5')
        
        print("모델 학습 완료!")
        return True
        
    except Exception as e:
        print(f"모델 학습 중 오류 발생: {str(e)}")
        return False 